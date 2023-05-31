import base64
import json
import psutil
import nvgpu
import pynvml
from nvgpu import list_gpus

import torch
import cv2
import numpy as np
import pyiqa
from piqa import PSNR, SSIM, LPIPS

from torchvision.utils import save_image
from ts.torch_handler.base_handler import BaseHandler

from ts.service import emit_metrics
from ts.metrics.dimension import Dimension
from ts.metrics.metric_type_enum import MetricTypes
import ts.metrics.system_metrics as system_metrics

"""
ModelHandler defines a custom model handler.
"""

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self.context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.map_location = None
        self.device = None
        self.use_gpu = True
        self.realesrgan_model = None
        self.metrics = None
        self.height = None
        self.width = None
        self.niqe_old = None
        self.brisque_old = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.context = context
        properties = context.system_properties
        gpu_id = properties.get("gpu_id")

        self.map_location, self.device, self.use_gpu = \
            ("cuda", torch.device("cuda:"+str(gpu_id)), True) if torch.cuda.is_available() else \
            ("cpu", torch.device("cpu"), False)

        self.metrics = self.context.metrics

        #  load the model, refer 'custom handler class' above for details
        from model import RRDBNet
        self.realesrgan_model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=23, 
            num_grow_ch=32, 
            scale=4)

        #state_dict = torch.load('RealESRGAN_x4plus.pth')
        state_dict = torch.load('net_g_75000.pth')
        param_key_g = "params_ema"
        self.realesrgan_model.load_state_dict(state_dict[param_key_g] if param_key_g in state_dict.keys() else state_dict, strict=True)
        self.realesrgan_model.to(self.device)
        self.realesrgan_model.eval()
        
        self.initialized = True

    def preprocess(self, requests):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        
        # Take the input data and make it inference ready
        for req in requests:
            img = req.get("data")
            if img is None:
                img = req.get("body")
        
        self.metrics.add_size('SizeOfImage', len(img) / 1024, None, 'kB')
        np_img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        img_lr = img.astype(np.float32) / 255.0
        
        # System Metrics during preprocess
        self.height, self.width = img_lr.shape[:2]
        self.metrics.add_counter('ImageHeight', self.height)
        self.metrics.add_counter('ImageWidth', self.width)
        self.cpu_metrics()
        
        # BGR to RGB
        # Set input image as Tensor object
        img_lr = np.transpose(img_lr if img_lr.shape[2] == 1 else img_lr[:, :, [2, 1, 0]], (2, 0, 1))
        img_lr = torch.from_numpy(img_lr).float().unsqueeze(0).to(self.device)
        
        # Capture NIQE/BRISQUE metric scores
        # self.niqe_old, self.brisque_old = self.no_ref_metrics(img_lr)

        # GPU Metrics during preprocess
        self.gpu_metrics(torch.cuda.device_count())
    
        return img_lr
    
    def inference(self, img_lr):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # System Metrics before inference
        self.cpu_metrics()
        # GPU Metrics before inference
        self.gpu_metrics(torch.cuda.device_count())
        
        with torch.no_grad():
            output = self.realesrgan_model(img_lr)
            output = output.to(self.device)

        # System Metrics after inference
        self.cpu_metrics()
        # GPU Metrics after inference
        self.gpu_metrics(torch.cuda.device_count())

        return output

    def postprocess(self, output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Capture NIQE/BRISQUE score of resulting image
        # niqe_new, brisque_new = self.no_ref_metrics(output)

        # Take output from network and post-process to desired format
        # Save Image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            # CHW-RGB to HCW-BGR
            output = np.transpose(output[[2,1,0], :, :], (1,2,0))

        # Obtain Image Resolution
        height, width = output.shape[:2]
        self.metrics.add_counter('ImageHeight', height)
        self.metrics.add_counter('ImageWidth', width)

        # Convert float32 to uint8
        output = (output * 255.0).round().astype(np.uint8)
        
        # Write result as .png image file
        _, img_encoded = cv2.imencode('.png', output)

        self.metrics.add_size('SizeOfImage', len(img_encoded.tobytes()) / 1024, None, 'kB')
        # System Metrics after postprocessing
        self.cpu_metrics()
        # GPU Metrics after postprocessing
        self.gpu_metrics(torch.cuda.device_count())

        # Convert output to base64-encoded string
        img_bytes = img_encoded.tobytes()
        # img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # Serialize NIQE/BRISQUE scores
        # niqe_old_str = str(self.niqe_old)
        # niqe_new_str = str(niqe_new)
        # brisque_old_str = str(self.brisque_old)
        # brisque_new_str = str(brisque_new)

        # Construct the JSON response
        # response = {'result': img_base64, 'niqe_old': niqe_old_str, 'niqe_new': niqe_new_str, 
        #             'brisque_old': brisque_old_str, 'brisque_new': brisque_new_str}
        # response_bytes = json.dumps(response).encode('utf-8')
        
        # return [response_bytes]
        return [img_bytes]


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)

    def cpu_metrics(self):
        """
        CPU Metrics to capture during inference.
        """
        self.metrics.add_percent('CPUUtilization', psutil.cpu_percent())
        self.metrics.add_size('DiskAvailable', psutil.disk_usage("/").free / (1024 * 1024 * 1024))
        self.metrics.add_size('DiskUsage', psutil.disk_usage("/").used / (1024 * 1024 * 1024))
        self.metrics.add_percent('DiskUtilization', psutil.disk_usage("/").percent)
        self.metrics.add_size('MemoryAvailable', psutil.virtual_memory().available / (1024 * 1024))
        self.metrics.add_size('MemoryUsed', psutil.virtual_memory().used / (1024 * 1024))
        self.metrics.add_percent('MemoryUtilization', psutil.virtual_memory().percent)

    def gpu_metrics(self, num_of_gpu):
        """
        GPU Metrics to capture during inference, if GPU exists.
        """
        if num_of_gpu <= 0:
            return
        
        info = nvgpu.gpu_info()
        for value in info:
            dimension_gpu = [
                Dimension('Level', 'Host'),
                Dimension('device_id', value['index']),
            ]
            self.metrics.add_percent("GPUMemoryUtilization", value['mem_used_percent'], 'percent', dimension_gpu)
            self.metrics.add_size("GPUMemoryUsed", value['mem_used'], unit='MB', dimensions=dimension_gpu)
        
        try:
            statuses = list_gpus.device_statuses()
        except pynvml.nvml.NVMLError_NotSupported:
            statuses = []

        for idx, value in enumerate(statuses):
            dimension_gpu = [
                Dimension("Level", "Host"), 
                Dimension("device_id", idx),
            ]
            self.metrics.add_percent("GPUUtilization", value['utilization'], 'percent', dimension_gpu)

    def no_ref_metrics(self, image):
        """
        Image quality metrics, NIQE and BRISQUE to capture during inference.
        """
        niqe_metric = pyiqa.create_metric('niqe', device=self.device)
        brisque_metric = pyiqa.create_metric('brisque', device=self.device)

        niqe_score = niqe_metric(image)
        brisque_score = brisque_metric(image)
        
        self.metrics.add_percent('NIQEScore', niqe_score.item())
        self.metrics.add_percent('BRISQUEScore', brisque_score.item())
        
        return [niqe_score.item(), brisque_score.item()]

        
