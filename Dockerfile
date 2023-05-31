FROM nvidia/cuda:10.2-base-ubuntu18.04
WORKDIR /usr/src/sr_training

FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
USER root

RUN apt-get update --fix-missing \
&& apt-get install -y zip unzip curl wget vim \
python3-pip

COPY sr_part2/datasets/FFHQ-69000.zip ./
COPY sr_part2/datasets/SCUT.zip ./

COPY sr_part2/requirements.txt ./
RUN pip install -r requirements.txt

# Add NVIDIA container toolkit repository
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
     && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
     && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list \
     && apt-get update

# Install NVIDIA container toolkit
RUN apt-get install -y nvidia-container-toolkit

# Remove residual installation files
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*