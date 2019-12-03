 <img src="https://github.com/yuanming-hu/taichi/raw/master/misc/logo.png">

### Build
| **Linux (CUDA)** | **OS X** | **Windows** |
|:-----|:-----|:-----|
|[![Build Status](http://f11.csail.mit.edu:8080/job/taichi/badge/icon)](http://f11.csail.mit.edu:8080/job/taichi/)|[![Build Status](https://travis-ci.com/yuanming-hu/taichi.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi) | [![Build Status](https://ci.appveyor.com/api/projects/status/github/yuanming-hu/taichi?branch=master&svg=true)](https://ci.appveyor.com/project/IteratorAdvance/taichi) |

### Python Wheels
| **Linux** | **OS X** | **Windows** |
|:-----|:-----|:-----|
|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build status](https://ci.appveyor.com/api/projects/status/39ar9wa8yd49je7o?svg=true)](https://ci.appveyor.com/project/IteratorAdvance/taichi-wheels-test)|

### Misc.
| **Documentation** | **Chat** |
|:-----|:-----|
| [![Documentation Status](https://readthedocs.org/projects/taichi/badge/?version=latest)](http://taichi.readthedocs.io/en/latest/?badge=latest) | [![Join the chat at https://gitter.im/taichi-dev/Lobby](https://badges.gitter.im/taichi-dev/Lobby.svg)](https://gitter.im/taichi-dev/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |

```bash
# CPU only. No GPU/CUDA needed. (Linux, OS X and Windows)
python3 -m pip install taichi-nightly

# With GPU (CUDA 10.0) support (Linux only)
python3 -m pip install taichi-nightly-cuda-10-0

# With GPU (CUDA 10.1) support (Linux only)
python3 -m pip install taichi-nightly-cuda-10-1
```

### High-Performance Computation on Sparse Data Structures [[Paper]](http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[Language Details]](https://github.com/yuanming-hu/taichi/blob/master/python/taichi/README.md) [[Taichi Compiler Developer Installation]](https://taichi.readthedocs.io/en/latest/dev_install.html)

## [Examples](https://github.com/yuanming-hu/taichi/tree/master/examples)
## Updates
- (Dec  3, 2019) v0.2.1 released.
   - Improved type mismatch error message
   - native `min`/`max` supprt
   - Tensor access index dimensionality checking
   - `Matrix.to_numpy`, `Matrix.zero`, `Matrix.identity`, `Matrix.fill`
   - Warning instead of error on lossy stores
   - Added some initial support for cross-referencing local variables in different offloaded blocks.
- (Nov 28, 2019) v0.2.0 released.
   - More friendly syntax error when passing non-compile-time-constant values to `ti.static`
   - Systematically resolved the variable name resolution [issue](https://github.com/yuanming-hu/taichi/issues/282)
   - Better interaction with numpy:
     - `numpy` arrays passed as a `ti.ext_arr()` [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_numpy.py)
       - `i32/f32/i64/f64` data type support for numpy
       - Multidimensional numpy arrays now supported in Taichi kernels 
     - `Tensor.to_numpy()` and `Tensor.from_numpy(numpy.ndarray)` supported [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_cvt_numpy.py)
     - Corresponding PyTorch tensor interaction will be supported very soon. Now only 1D f32 PyTorch tensors supproted when using `ti.ext_arr()`. Please use numpy arrays as intermediate buffers for now
   - Indexing arrays with an incorrect number of indices now results in a syntax error
   - Tensor shape reflection: [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_tensor_reflection.py)
     - `Tensor.dim()` to retrieve the dimensionality of a global tensor
     - `Tensor.shape()` to retrieve the shape of a global tensor
     - Note the above queries will cause data structures to be materialized
   - `struct-for` (e.g. `for i, j in x`) now supports iterating over tensors with non power-of-two dimensions
   - Handy tensor filling: [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_fill.py)
     - `Tensor.fill(x)` to set all entries to `x`
     - `Matrix.fill(x)` to set all entries to `x`, where `x` can be a scalar or `ti.Matrix` of the same size 
   - Reduced python package size
   - `struct-for` with grouped indices for better metaprogramming, especially in writing dimensionality-independent code, in e.g. physical simulation: [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_grouped.py)
```python
for I in ti.grouped(x): # I is a vector of size x.dim() and data type i32
  x[I] = 0
  
# If tensor x is 2D 
for I in ti.grouped(x): # I is a vector of size x.dim() and data type i32
  y[I + ti.Vector([0, 1])] = I[0] + I[1]
# is equivalent to
for i, j in x:
  y[i, j + 1] = i + j
```
     
- (Nov 27, 2019) v0.1.5 released. 
   - [Better modular programming support](https://github.com/yuanming-hu/taichi/issues/282)
   - Disalow the use of `ti.static` outside Taichi kernels
   - Documentation improvements (WIP)
   - Codegen bug fixes
   - Special thanks to Andrew Spielberg and KLozes for bug report and feedback.
- (Nov 22, 2019) v0.1.3 released. 
   - Object-oriented programming. [[Example]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_oop.py)
   - native Python function translation in Taichi kernels: 
     - Use `print` instead of `ti.print`
     - Use `int()` instead of `ti.cast(x, ti.i32)` (or `ti.cast(x, ti.i64)` if your default integer precision is 64 bit)
     - Use `float()` instead of `ti.cast(x, ti.f32)` (or `ti.cast(x, ti.f64)` if your default float-point precision is 64 bit)
     - Use `abs` instead of `ti.abs`
     - Use `ti.static_print` for compile-time printing
     
- (Nov 16, 2019) v0.1.0 released. Fixed PyTorch interface. 
- (Nov 12, 2019) v0.0.87 released.
   - Added experimental Windows support with a [[known issue]](https://github.com/yuanming-hu/taichi/issues/251) regarding virtual memory allocation, which will potentially limit the scalability of Taichi programs (If you are a Windows expert, please let me know how to solve this. Thanks!). Most examples work on Windows now.
   - CUDA march autodetection;
   - [Complex kernel](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_complex_kernels.py) to override autodiff.
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
