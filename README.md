# The Taichi Programming Language [[Details]](https://github.com/yuanming-hu/taichi/blob/master/src/README.md)

| **Linux (CUDA)** | **OS X** |**Chat** |
|-------|-----|------|
|[![Build Status](http://f11.csail.mit.edu:8080/job/taichi/badge/icon)](http://f11.csail.mit.edu:8080/job/taichi/)|[![Build Status](https://travis-ci.com/yuanming-hu/taichi.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi)|[![Join the chat at https://gitter.im/taichi-dev/Lobby](https://badges.gitter.im/taichi-dev/Lobby.svg)](https://gitter.im/taichi-dev/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |

### High-Performance Computation on Sparse Data Structures [[Paper]](http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo)

## Updates
```bash
# CPU only. No GPU/CUDA needed
python3 -m pip install taichi-nightly

# With GPU (CUDA 10.0) support
python3 -m pip install taichi-nightly-cuda-10-0

# With GPU (CUDA 10.1) support
python3 -m pip install taichi-nightly-cuda-10-1
```

 - (Oct 24, 2019) Python wheels (v0.0.61) released for Python 3.6/3.7 and CUDA 10.0/10.1 on Ubuntu 16.04+. Contributors of this release include *Yuanming Hu, robbertvc, Zhoutong Zhang, Tao Du, Srinivas Kaza, and Kenneth Lozes*.
 - (Oct 22, 2019) Added support for [kernel templates](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_kernel_templates.py). Kernel templates allow users to pass in taichi tensors and compile-time constants as kernel parameters.
 - (Oct 9, 2019) Compatibility improvements. Added a basic PyTorch interface. [[Example]](https://github.com/yuanming-hu/taichi/blob/master/examples/torch_tensor_ad.py).
 - (Oct 7, 2019) Released **experimental** python **3.6** wheels on Linux (tested on Ubuntu 16.04/18.04) for those who are eager to try without building from source. More stable releases are coming in a few days. To install them: 

Notes: 
   - You still need to clone this repo for demo scripts under `examples`. You *do not* need to execute `install.py`.
   After installation using `pip` you can simply go to `examples` and execute, e.g., `python3 mpm.py`.
   - Make sure you have `clang-7`. On Ubuntu 18.04 you can install it with `sudo apt-get install clang-7`. See [here](https://askubuntu.com/questions/1113974/using-c17-with-clang-on-ubuntu-16-04) for installing `clang-7` on Ubuntu 16.04. You will also need `g++-7` on Ubuntu16.04 (for c++17 headers). To install:
   ```bash
# Ubuntu 16.04 only
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-7 -y
   ```
   - Make sure you clear your legacy Taichi installation (if applicable) by cleaning the environment variables (delete `TAICHI_REPO_DIR`, and remove legacy taichi from `PYTHONPATH`) in your `.bashrc` or `.zshrc`. Or you can simply do this in your shell to temporarily clear them:
   ```
   export PYTHONPATH=
   export TAICHI_REPO_DIR=
   ```

<!---
| **Linux, Mac OS X** | **Windows** | Doc (WIP) | **Chat** |
|---------------------|------------------|----------------|------------------|
|[![Build Status](https://travis-ci.org/yuanming-hu/taichi.svg?branch=master)](https://travis-ci.org/yuanming-hu/taichi)|[![Build Status](https://ci.appveyor.com/api/projects/status/github/yuanming-hu/taichi?branch=master&svg=true)](https://ci.appveyor.com/project/IteratorAdvance/taichi)|[![Documentation Status](https://readthedocs.org/projects/taichi/badge/?version=latest)](http://taichi.readthedocs.io/en/latest/?badge=latest)|[![Join the chat at https://gitter.im/taichi-dev/Lobby](https://badges.gitter.im/taichi-dev/Lobby.svg)](https://gitter.im/taichi-dev/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)|
--->


#  The Taichi Library [[Legacy branch]](https://github.com/yuanming-hu/taichi/tree/legacy)
**Taichi** is an open-source computer graphics library that aims to provide easy-to-use infrastructures for computer graphics R&D. It's written in C++14 and wrapped friendly with Python.

## News
 - May 17, 2019: [Giga-Voxel SPGrid Topology Optimization Solver](https://github.com/yuanming-hu/spgrid_topo_opt) is released!
 - March 4, 2019: [MLS-MPM/CPIC solver](https://github.com/yuanming-hu/taichi_mpm) is now MIT-licensed!
 - August 14, 2018: [MLS-MPM/CPIC solver](https://github.com/yuanming-hu/taichi_mpm) reloaded! It delivers 4-14x performance boost over the previous state of the art on CPUs.
 
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/topopt/bird-beak.gif">

 <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/water_wheel.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/sand_paddles.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/armodillo.gif" style=""> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/debris_flow.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/sand-sweep.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/sand_stir.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/bunny.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/robot_forward.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/banana.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/cheese.gif">
 
## [Getting Started](https://taichi.readthedocs.io/en/latest/installation.html#)
