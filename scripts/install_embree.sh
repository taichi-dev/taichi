#!/bin/bash
wget https://github.com/embree/embree/releases/download/v2.13.0/embree-2.13.0.x86_64.rpm.tar.gz
tar xzf embree-2.13.0.x86_64.rpm.tar.gz
sudo alien embree-lib-2.13.0-1.x86_64.rpm
sudo alien embree-devel-2.13.0-1.x86_64.rpm
sudo alien embree-examples-2.13.0-1.x86_64.rpm
sudo dpkg -i embree-lib_2.13.0-2_amd64.deb
sudo dpkg -i embree-devel_2.13.0-2_amd64.deb
sudo dpkg -i embree-examples_2.13.0-2_amd64.deb
