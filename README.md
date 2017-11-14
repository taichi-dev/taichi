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

# Installation
## Step 1: Install Prerequists
### On Ubuntu 16.04

```
sudo apt-get install python3 git build-essential cmake make g++ libtbb-dev alien dpkg-dev debhelper ffmpeg python3-tk python3-pip
```
Install `embree` and `tbb`:
```
cd build
sudo sh ../install_embree.sh
```

### On Arch Linux
```
sudo pacman -S python3 python-pip make cmake intel-tbb embree ffmpeg tk
```

## Step 2:
Append to your `~/.bashrc`:
```
export TAICHI_ROOT_DIR=/home/yuanming/repos/                  
export PYTHONPATH=$PYTHONPATH:$TAICHI_ROOT_DIR/taichi/python
```

## Step 3:
Start `taichi`: (this will automatically install required python packages and build `taichi`.)
```shell
$ python3
```
```python3
>>> import taichi as tc
>>> ...
```

(Research Projects should be put into the folder `projects`, and will be automatically detected if the folder contains a `CMakeLists.txt`.)

# View results
```
python3 python/examples/server/main.py 
```
Then open page http://localhost:1111. Choose a simulation output, press `P` to play, `R` to change frame rate (faster/slower).

# Examples
Please see [examples](https://github.com/yuanming-hu/taichi/tree/dev/python/examples).

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
 
Note that all of them are bundled in `taichi` (in source (header)/binary forms), and users do not have to manually install them.
