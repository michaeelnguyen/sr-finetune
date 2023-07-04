# Deep Learning-based Super-Resolution on the Cloud: Focus on Face and Text Enhancement
This experiment expands upon my [Master's thesis](https://doi.org/10.5281/zenodo.7897859), which used pre-trained SR models for inferencing. The experiment attempts to fine-tune the pre-trained models for face and text image enhancement using the Flickr-Faces-HQ dataset (FFHQ) (last 1000 images) and SCUT-CTW1500 (first 100 images), a curved text dataset.
 
A Google Drive link is provided showing the Image Quality Assessment (IQA) and visual results from inferencing the pre-trained and finetuned Real-ESRGAN and SwinIR Super-Resolution models with image benchmarks datasets, Set5 and Set14, and the Vimeo-90K test set, a video Super-Resolution dataset of video frames extracted as images. The trained models (.pth) are also stored here. [Link](https://drive.google.com/drive/folders/1HnPhSydsVox-Ds2FI-l06Jstp2XGxTvw?usp=sharing)

# Abstract
Real-ESRGAN and SwinIR are two deep learning models for Single-Image Super-Resolution (SISR), which attempt to address real-world scenarios for image enhancement. However, the pre-trained models do not effectively handle LR images containing human faces and text. An experiment is conducted to expand upon the training performed in their respective studies and improve the image enhancement using a cloud computing environment. Traditional image quality metrics, Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity (SSIM), are used to objectively evaluate the image quality. Three learning-based perceptual metrics, the Blind / Referenceless Image Spatial Quality Evaluator (BRISQUE), Naturalness Image Quality Evaluator (NIQE), and the Learned Perceptual Image Patch Similarity (LPIPS), are also incorporated to assess how images would be subjectively perceived based on human perception. To evaluate the model performance of Real-ESRGAN and SwinIR, specifically for face and text images, both traditional and perceptual metrics are taken into consideration, in addition to the cost associated with model training using Microsoft Azure. The findings show that with additional fine-tuning, SwinIR has slightly improved PSNR and SSIM values while taking less training time compared to Real-ESRGAN at the cost of perceptual quality.

## References
    @article{wang2021realesrgan,
      title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data}, 
      author={Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
      journal={2107.10833},
      year={2021},
    }
    @article{liang2021swinir,
      title={SwinIR: Image Restoration Using Swin Transformer},
      author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
      journal={arXiv preprint arXiv:2108.10257},
      year={2021}
    }
## License and Acknowledgement
The codes are based on [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) under BSD-3 License, [SwinIR](https://github.com/JingyunLiang/SwinIR) under Apache 2.0 License, and [KAIR](https://github.com/cszn/KAIR) under MIT License.
