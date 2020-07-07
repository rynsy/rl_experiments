FROM tensorflow/tensorflow:1.15.0-py3

RUN export TZ=America/New_York
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN ["apt-get", "install", "-y", "build-essential", "libopencv-dev"]
RUN ["apt-get", "install", "-y", "git", "libsm6", "libxext6", "libxrender-dev", "libopenmpi-dev"]
RUN ["apt-get", "install", "-y", "--fix-missing", "python3-opengl"]

WORKDIR /workspace
COPY . /workspace
RUN pip3 install --upgrade pip
RUN pip3 install -r /workspace/requirements.txt
#RUN pip3 install git+git://github.com/openai/spinningup.git
RUN python3 ddpg.py
