FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y libgl1-mesa-dev \
    && apt-get install -y libglib2.0 
    # && pip install --upgrade pip \
    # && pip install setuptools

RUN apt install -y curl python3 python3-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py
COPY ./requirements.txt /code/requirements.txt
WORKDIR /code

RUN pip install -r requirements.txt