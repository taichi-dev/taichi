FROM ubuntu:16.04

RUN mkdir -p /home/travis/
RUN apt-get update
RUN apt-get install -y python3
RUN python3 install.py