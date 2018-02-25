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
python 3.5+ is required.
```
wget https://raw.githubusercontent.com/yuanming-hu/taichi/master/install.py && python3 install.py
```
Suppoprted Platforms (open an issue if you find the script fails):
 - Ubuntu (gcc 5+)
 - Mac OS X (gcc 5+, clang 4.0+)
 - Windows (Microsoft Visual Studio 2017)

# View results
```
python3 $TAICHI_ROOT_DIR/taichi/python/examples/server/main.py 
```
Then open page http://localhost:1111. Choose a simulation output, press `P` to play, `R` to change frame rate (faster/slower).

# Examples
Please see [examples](https://github.com/yuanming-hu/taichi/tree/master/projects/examples).

## Acknowledgements

Like many other open-source projects, `taichi` is based on other open-source projects, which are shipped with taichi and users do not have to install manually:
 - [Intel Embree](https://embree.github.io/)
 - [Intel TBB](https://www.threadingbuildingblocks.org/)
 - [fmt](https://github.com/fmtlib/fmt)
 - [Catch2](https://github.com/catchorg/Catch2)
 - [spdlog](https://github.com/gabime/spdlog)
 - [stb_image, stb_image_write, stb_truetype](https://github.com/nothings/stb)
 - [tinyobjloader](https://github.com/syoyo/tinyobjloader)
 - [JIXIE::ImplicitQRSVD](http://www.math.ucla.edu/~fuchuyuan/svd/paper.pdf)
 - [dcraw](https://www.cybercom.net/~dcoffin/dcraw/)
 - [ffmpeg](https://www.ffmpeg.org/)
 - [pybind11](https://github.com/pybind/pybind11)
 - ...

Current developers include:
 - [Yuanming Hu](http://taichi.graphics/me) (Project creator & main developer. MIT CSAIL, Ph.D. student (1st year). )
 - [Yu Fang](http://squarefk.com/) (Developer. Tsinghua University, senior undergraduate student. Next: Ph.D. student at University of Pennsylvania with [Prof. Chenfanfu Jiang](http://www.seas.upenn.edu/~cffjiang/))
 - ...