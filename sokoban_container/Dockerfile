FROM pytorch/pytorch:latest

RUN export TZ=America/New_York
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y build-essential libopencv-dev 
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]
RUN ["apt-get", "install", "-y", "python3-opengl"]

WORKDIR /workspace
COPY . /workspace
RUN pip install -r /workspace/requirements.txt
