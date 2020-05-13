<div align="center">
  <img width="500px" src="https://github.com/yuanming-hu/taichi/raw/master/misc/logo.png">
  <h3> <a href="https://taichi.readthedocs.io/en/latest/hello.html"> Tutorial </a> | <a href="https://github.com/yuanming-hu/taichi/tree/master/examples"> Examples </a> | <a href="https://taichi.readthedocs.io/en/latest/contributor_guide.html"> Contributor Guidelines </a> | <a href="https://forum.taichi.graphics/"> Forum </a> </h3>
  <h3> <a href="https://taichi.readthedocs.io/en/stable/"> Documentation </a> | <a href="https://taichi.readthedocs.io/zh_CN/latest/"> 简体中文文档 </a> </h3>
</div>

[![Build Status](https://img.shields.io/travis/taichi-dev/taichi?logo=Travis)](https://travis-ci.com/taichi-dev/taichi)
[![Build Status](https://img.shields.io/appveyor/build/yuanming-hu/taichi?logo=AppVeyor)](https://ci.appveyor.com/project/yuanming-hu/taichi/branch/master)
[![Latest Release](https://img.shields.io/github/v/release/taichi-dev/taichi?color=blue)](https://github.com/taichi-dev/taichi/releases/latest)

## Overview

**Taichi** (太极) is a programming language designed for *high-performance computer graphics*. It is deeply embedded in **Python**, and its **just-in-time compiler** offloads compute-intensive tasks to multi-core CPUs and massively parallel GPUs.

<a href="https://github.com/taichi-dev/taichi/blob/master/examples/fractal.py#L1-L31"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/taichi/fractal_code.png" height="293px"></a>  <img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/fractal_small.gif" height="293px">


Advanced features of Taichi include [spatially sparse computing](https://taichi.readthedocs.io/en/latest/sparse.html) and [differentiable programming](https://taichi.readthedocs.io/en/latest/differentiable_programming.html) [[examples]](https://github.com/yuanming-hu/difftaichi).

## Gallery

<a href="https://github.com/taichi-dev/taichi/blob/master/examples/mpm128.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/mpm128.gif" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/stable_fluid.py"> <img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/stable_fluids.gif" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/sdf_renderer.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/sdf_renderer.jpg" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/taichi_sparse.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/sparse_grids.gif" height="192px"></a>

<a href="https://github.com/taichi-dev/taichi/blob/master/examples/mpm_lagrangian_forces.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/lagrangian.gif" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/pbf.gif" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/game_of_life.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/game_of_life.gif" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/euler.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/euler.gif" height="192px"></a>

## Installation [![Downloads](https://img.shields.io/pypi/dw/taichi)](https://pepy.tech/project/taichi)

```bash
python3 -m pip install taichi
```

**Supported OS**: Windows, Linux, Mac OS X; **Python**: 3.6, 3.7, 3.8; **Backends**: x64 CPUs, CUDA, Apple Metal, OpenGL Compute Shaders.

Please build from source for other configurations (e.g., your CPU is ARM).

**Note:**
 - Starting April 13 2020 (v0.5.12), we release the Python package [`taichi`](https://pypi.org/project/taichi/) instead of [`taichi-nightly`](https://pypi.org/project/taichi-nightly/). Now this PyPI package includes CPU, CUDA 10, Metal and OpenGL support.
 - On Ubuntu 19.04+, please `sudo apt install libtinfo5`.
 - On Windows, please install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) if you haven't.
 - [[All releases]](https://github.com/taichi-dev/taichi/releases) [[Change log]](misc/changelog.md)

|| **Linux (CUDA)** | **OS X (10.14+)** | **Windows** | **Documentation**|
|:------|:-----|:-----|:-----|:-----|
|**Build**|[![Build Status](http://f11.csail.mit.edu:8080/job/taichi/badge/icon)](http://f11.csail.mit.edu:8080/job/taichi/)| [![Build Status](https://travis-ci.com/taichi-dev/taichi.svg?branch=master)](https://travis-ci.com/taichi-dev/taichi) | [![Build status](https://ci.appveyor.com/api/projects/status/yxm0uniin8xty4j7/branch/master?svg=true)](https://ci.appveyor.com/project/yuanming-hu/taichi/branch/master)| [![Documentation Status](https://readthedocs.org/projects/taichi/badge/?version=latest)](http://taichi.readthedocs.io/en/latest/?badge=latest)|
|**PyPI**|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build status](https://ci.appveyor.com/api/projects/status/39ar9wa8yd49je7o?svg=true)](https://ci.appveyor.com/project/yuanming-hu/taichi-wheels-test) |

## Applications

- A High-Performance Multi-Material Continuum Physics Engine based on Taichi: [Taichi Elements](https://github.com/taichi-dev/taichi_elements) (work in progress)

## Developers

The Taichi project was created by [Yuanming Hu (yuanming-hu)](https://github.com/yuanming-hu). Significant contributions are made by:
 - [Ye Kuang (k-ye)](https://github.com/k-ye) (Apple Metal backend)
 - [彭于斌 (archibate)](https://github.com/archibate) (OpenGL Compute Shader backend)
 - [Mingkuan Xu (xumingkuan)](https://github.com/xumingkuan) (IR optimization & standardization)

[Kenneth Lozes (KLozes)](https://github.com/KLozes) and [Yu Fang (squarefk)](https://github.com/squarefk) have also made notable contributions. [[List of all contributors]](https://github.com/taichi-dev/taichi/graphs/contributors)

We welcome feedback and comments. If you would like to contribute to Taichi, please check out our [Contributor Guidelines](https://taichi.readthedocs.io/en/latest/contributor_guide.html).

If you use Taichi in your research, please cite our papers:

- [**(SIGGRAPH Asia 2019) Taichi: High-Performance Computation on Sparse Data Structures**](http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[BibTex]](https://raw.githubusercontent.com/yuanming-hu/taichi/master/misc/taichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/taichi)
- [**(ICLR 2020) DiffTaichi: Differentiable Programming for Physical Simulation**](https://arxiv.org/abs/1910.00935) [[Video]](https://www.youtube.com/watch?v=Z1xvAZve9aE) [[BibTex]](https://raw.githubusercontent.com/yuanming-hu/taichi/master/misc/difftaichi_bibtex.txt) [[Code]](https://github.com/yuanming-hu/difftaichi)
