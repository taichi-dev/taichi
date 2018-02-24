FROM ubuntu:16.04

RUN echo $PWD
RUN printenv
RUN ls
RUN cd yuanming-hu/taichi
RUN apt-get update
RUN apt-get install -y python3 wget sudo
RUN python3 install.py ci
