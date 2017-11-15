FROM ubuntu:16.04

RUN mkdir -p /home/travis/
RUN apt-get update
RUN apt-get install -y python3 wget sudo
RUN wget https://raw.githubusercontent.com/yuanming-hu/taichi/dev/install.py && python3 install.py ci
