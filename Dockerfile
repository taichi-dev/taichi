FROM ubuntu:16.04

RUN mkdir -p /home/travis/
RUN apt-get update
RUN apt-get install -y python3
RNU wget https://raw.githubusercontent.com/yuanming-hu/taichi/dev/install.py && python3 install.py
