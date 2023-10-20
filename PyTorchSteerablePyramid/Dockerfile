FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel 

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y libgl1-mesa-dev \
    && apt-get install -y libglib2.0-0 \
    && pip install --upgrade pip \
    && pip install setuptools

WORKDIR /home/work

COPY requirements.txt /home/work

RUN pip install -r requirements.txt