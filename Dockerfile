FROM ubuntu:16.04

RUN echo $PWD
RUN apt-get update
RUN apt-get install -y python3 wget sudo
python3 install.py ci

