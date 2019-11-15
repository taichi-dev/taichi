 <img src="https://github.com/yuanming-hu/taichi/raw/master/misc/logo.png">

| **Linux (CUDA)** | **OS X** | **Linux & OS X Wheels** |
|:-----|:-----|:-----|
|[![Build Status](http://f11.csail.mit.edu:8080/job/taichi/badge/icon)](http://f11.csail.mit.edu:8080/job/taichi/)|[![Build Status](https://travis-ci.com/yuanming-hu/taichi.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi) | [![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test) |
| **Windows** | **Windows Wheels** | **Chat** |
|[![Build Status](https://ci.appveyor.com/api/projects/status/github/yuanming-hu/taichi?branch=master&svg=true)](https://ci.appveyor.com/project/IteratorAdvance/taichi) |[![Build status](https://ci.appveyor.com/api/projects/status/39ar9wa8yd49je7o?svg=true)](https://ci.appveyor.com/project/IteratorAdvance/taichi-wheels-test)|[![Join the chat at https://gitter.im/taichi-dev/Lobby](https://badges.gitter.im/taichi-dev/Lobby.svg)](https://gitter.im/taichi-dev/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |

```bash
# CPU only. No GPU/CUDA needed. (Linux, OS X and Windows)
python3 -m pip install taichi-nightly

# With GPU (CUDA 10.0) support (Linux only)
python3 -m pip install taichi-nightly-cuda-10-0

# With GPU (CUDA 10.1) support (Linux only)
python3 -m pip install taichi-nightly-cuda-10-1
```

### High-Performance Computation on Sparse Data Structures [[Paper]](http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[Language Details]](https://github.com/yuanming-hu/taichi/blob/master/python/taichi/README.md) [[Taichi Compiler Developer Installation]](https://github.com/yuanming-hu/taichi/blob/master/taichi/README.md#developer-installation)

## [Examples](https://github.com/yuanming-hu/taichi/tree/master/examples)
## Updates
 - (Nov 12, 2019) v0.0.87 released.
   - Added experimental Windows support with a [[known issue]](https://github.com/yuanming-hu/taichi/issues/251) regarding virtual memory allocation, which will potentially limit the scalability of Taichi programs (If you are a Windows expert, please let me know how to solve this. Thanks!). Most examples run on Windows.
   - CUDA march autodetection;
   - [Complex kernel]()https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_complex_kernels.py to override autodiff.
 - (Nov 4, 2019) v0.0.85 released.
   - `ti.stop_grad` for stopping gradients during backpropagation. [[Example]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_stop_grad.py#L75);
   - Compatibility improvements on Linux and OS X;
   - Minor bug fixes.
- (Nov 1, 2019) v0.0.77 released.
   - **Python wheels now support OS X 10.14+**;
   - LLVM is now the default backend. No need to install `gcc-7` or `clang-7` anymore. To use legacy backends, ```export TI_LLVM=0```;
   - LLVM compilation speed is improved by 2x;
   - More friendly syntax error messages.
 - (Oct 30, 2019) v0.0.72 released.
   - LLVM GPU backend now as fast as the legacy (yet optimized) CUDA backend. To enable, ```export TI_LLVM=1```;
   - Bug fixes: LLVM `struct for` list generation.
 - (Oct 29, 2019) v0.0.71 released. LLVM GPU backend performance greatly improved. Frontend compiler now emits readable syntax error messages.
 - (Oct 28, 2019) v0.0.70 released. This version comes with experimental LLVM backends for x86_64 and CUDA (via NVVM/PTX). GPU kernel compilation speed is improved by 10x. To enable, update the taichi package and ```export TI_LLVM=1```.
 - (Oct 24, 2019) Python wheels (v0.0.61) released for Python 3.6/3.7 and CUDA 10.0/10.1 on Ubuntu 16.04+. Contributors of this release include *Yuanming Hu, robbertvc, Zhoutong Zhang, Tao Du, Srinivas Kaza, and Kenneth Lozes*.
 - (Oct 22, 2019) Added support for [kernel templates](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_kernel_templates.py). Kernel templates allow users to pass in taichi tensors and compile-time constants as kernel parameters.
 - (Oct 9, 2019) Compatibility improvements. Added a basic PyTorch interface. [[Example]](https://github.com/yuanming-hu/taichi/blob/master/examples/pytorch_tensor_ad.py).

Notes: 
   - You still need to clone this repo for demo scripts under `examples`. You *do not* need to execute `install.py` or `dev_setup.py`.
   After installation using `pip` you can simply go to `examples` and execute, e.g., `python3 mpm_fluid.py`.
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
 
## [Getting Started (Legacy)](https://taichi.readthedocs.io/en/latest/installation.html#)
