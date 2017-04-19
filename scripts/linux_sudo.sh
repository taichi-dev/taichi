apt-get install cmake libtbb-dev alien dpkg-dev debhelper freeglut3-dev python-scipy python-tk python-imaging-tk python-pip 
pip install numpy future futures watchdog Pillow scipy pybind11 pyglet
wget https://github.com/embree/embree/releases/download/v2.13.0/embree-2.13.0.x86_64.rpm.tar.gz
tar xzf embree-2.13.0.x86_64.rpm.tar.gz
alien embree-lib-2.13.0-1.x86_64.rpm
alien embree-devel-2.13.0-1.x86_64.rpm
alien embree-examples-2.13.0-1.x86_64.rpm
dpkg -i embree-lib_2.13.0-2_amd64.deb
dpkg -i embree-devel_2.13.0-2_amd64.deb
dpkg -i embree-examples_2.13.0-2_amd64.deb
#add-apt-repository ppa:ubuntu-toolchain-r/test
#apt-get update
#apt-get install gcc-6 g++-6
