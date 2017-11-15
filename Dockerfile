FROM ubuntu:16.04

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update
RUN python3 install.py