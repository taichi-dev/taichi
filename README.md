<div align="center">
  <img src="https://github.com/yuanming-hu/taichi_assets/raw/master/demos/snow.gif" style="width: 200%; height: 200%"><br><br>
</div>

**Taichi** is a physically based computer graphics library, with various simulation
and rendering algorithms supported ([What's inside?](http://taichi.graphics/#features)). It's written in C++14 and wrapped friendly
with Python.

| **`Linux`, `Mac OS`** | **`Windows`** | **Chat** |
|---------------------|------------------|------------------|
|[![Build Status](https://travis-ci.org/yuanming-hu/taichi.svg?branch=master)](https://travis-ci.org/yuanming-hu/taichi)|[![Build Status](https://ci.appveyor.com/api/projects/status/github/yuanming-hu/taichi?branch=master&svg=true)](https://ci.appveyor.com/project/IteratorAdvance/taichi)|[![Join the chat at https://gitter.im/taichi-dev/Lobby](https://badges.gitter.im/taichi-dev/Lobby.svg)](https://gitter.im/taichi-dev/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)|

## Gallery [(More...)](http://taichi.graphics/gallery/)
![image](https://github.com/yuanming-hu/taichi_assets/raw/master/demos/smoke_cropped.gif)
![image](https://github.com/yuanming-hu/taichi_assets/raw/master/demos/microfacet.gif)
![image](https://github.com/yuanming-hu/taichi_assets/raw/master/demos/paper-cut.png)

## Getting Started

### Installation

#### Ubuntu 16.04
Packages:
https://github.com/yuanming-hu/config/wiki

```
sudo apt-get install python3 git build-essential g++ libtbb-dev alien dpkg-dev debhelper ffmpeg python3-tk python3-pip
sudo python3 -m pip install numpy Pillow scipy pybind11 flask flask_cors
```
Install `embree` and `tbb`:
```
cd build
sudo sh ../install_embree.sh
```

#### Arch Linux
```
sudo pacman -S python3 python-pip make cmake intel-tbb embree ffmpeg tk
```

####
Append to your `~/.bashrc`:
```
export TAICHI_ROOT_DIR=/home/yuanming/repos/                  
export PYTHONPATH=$PYTHONPATH:$TAICHI_ROOT_DIR/taichi/python
```

### View results
```
python3 python/examples/server/main.py 
```
and then open your browser `localhost:1111`. Choose a simulation output, press `P` to play, `R` to change frame rate (faster/slower).

(Research Projects should be put into the folder `projects`, and will be automatically detected if the folder contains a `CMakeLists.txt`.)

## Examples
Please see [examples](https://github.com/yuanming-hu/taichi/tree/dev/python/examples).

### Create your world with Taichi
```shell
$ python
```
```python
>>> import taichi as tc
>>> ...
```

## Acknowledgements

`Taichi`, like many other open-source projects, is based on other open-source projects:
 - [stb_image, stb_image_write, stb_truetype](https://github.com/nothings/stb)
 - [tinyobjloader](https://github.com/syoyo/tinyobjloader)
 - [fmt](https://github.com/fmtlib/fmt)
 - [spdlog](https://github.com/gabime/spdlog)
 - [Catch2](https://github.com/catchorg/Catch2)
 - [JIXIE::ImplicitQRSVD](http://www.math.ucla.edu/~fuchuyuan/svd/paper.pdf)
 - [dcraw](https://www.cybercom.net/~dcoffin/dcraw/)
 - [Intel Embree](https://embree.github.io/)
 - [Intel TBB](https://www.threadingbuildingblocks.org/)
 
Note that all of them (except `Intel Embree` and `TBB`) are head-only and bundled in `Taichi`, and users do not have to install them.
