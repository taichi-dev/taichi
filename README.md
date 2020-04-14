<div align="center">
  <img width="500px" src="https://github.com/yuanming-hu/taichi/raw/master/misc/logo.png">
  <h3> <a href="https://taichi.readthedocs.io/en/latest/"> Documentation </a> | <a href="https://taichi.readthedocs.io/en/latest/hello.html"> Tutorial </a> | <a href="https://github.com/yuanming-hu/difftaichi"> DiffTaichi </a> | <a href="https://github.com/yuanming-hu/taichi/tree/master/examples"> Examples </a> | <a href="https://taichi.readthedocs.io/en/latest/contributor_guide.html"> Contributor Guidelines </a> | <a href="https://forum.taichi.graphics/"> Forum </a> </h3>
</div>

## Overview

Taichi (太极) is a high-performance programming language and compiler, designed for computer graphics applications. It has native support for spatially sparse computation and differentiable programming.

<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/taichi/fractal_code.png" height="270px"> <img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/fractal.gif" height="270px">

## Gallery

<img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/sparse_grids.gif" height="192px"> <img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/stable_fluids.gif" height="192px">
<img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/sdf_renderer.jpg" height="192px">
<img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/mpm128.gif" height="192px">


## Installation [![Downloads](https://pepy.tech/badge/taichi-nightly)](https://pepy.tech/project/taichi-nightly)

```bash
python3 -m pip install taichi
```
**Supported OS**: Windows, Linux, Mac OS X; **Python**: 3.6, 3.7, 3.8; **Backends**: x64 CPUs, CUDA, Apple Metal.
 
Please build from source for other configurations (e.g., you need the experimental OpenGL backend or your CPU is ARM).

**Note:**
 - Starting April 13 2020 (v0.5.12), we release the Python package [`taichi`](https://pypi.org/project/taichi/) instead of [`taichi-nightly`](https://pypi.org/project/taichi-nightly/). This PyPI package has CPU, CUDA 10 and Metal support.
 - On Ubuntu 19.04+, please `sudo apt install libtinfo5`.
 - On Windows, please install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) if you haven't.
 - [[All releases]](https://github.com/taichi-dev/taichi/releases) [[Change log]](changelog.md)

|| **Linux (CUDA)** | **OS X (10.14+)** | **Windows** | **Documentation**|
|:------|:-----|:-----|:-----|:-----|
|**Build**|[![Build Status](http://f11.csail.mit.edu:8080/job/taichi/badge/icon)](http://f11.csail.mit.edu:8080/job/taichi/)| [![Build Status](https://travis-ci.com/taichi-dev/taichi.svg?branch=master)](https://travis-ci.com/taichi-dev/taichi) | [![Build status](https://ci.appveyor.com/api/projects/status/yxm0uniin8xty4j7/branch/master?svg=true)](https://ci.appveyor.com/project/yuanming-hu/taichi/branch/master)| [![Documentation Status](https://readthedocs.org/projects/taichi/badge/?version=latest)](http://taichi.readthedocs.io/en/latest/?badge=latest)|
|**PyPI**|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build status](https://ci.appveyor.com/api/projects/status/39ar9wa8yd49je7o?svg=true)](https://ci.appveyor.com/project/yuanming-hu/taichi-wheels-test) |

## Developers

The Taichi project was created by [Yuanming Hu (yuanming-hu)](https://github.com/yuanming-hu). Significant contributions are made by:
 - [Ye Kuang (k-ye)](https://github.com/k-ye) (Apple Metal backend)
 - [彭于斌 (archibate)](https://github.com/archibate) (OpenGL Compute Shader backend)
 - [Mingkuan Xu (xumingkuan)](https://github.com/xumingkuan) (IR optimization & standardization)
 
[KLozes](https://github.com/KLozes) and [Yu Fang (squarefk)](https://github.com/squarefk) have also made notable contributions. [[List of all contributors]](https://github.com/taichi-dev/taichi/graphs/contributors)

We welcome any feedback and comments. If you would like to contribute to Taichi, please check out our [Contributor Guidelines](https://taichi.readthedocs.io/en/latest/contributor_guide.html).

If you use `taichi` in your research, please cite our papers:

- [**(SIGGRAPH Asia 2019) Taichi: High-Performance Computation on Sparse Data Structures**](http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[BibTex]](https://raw.githubusercontent.com/yuanming-hu/taichi/master/misc/taichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/taichi)
- [**(ICLR 2020) DiffTaichi: Differentiable Programming for Physical Simulation**](https://arxiv.org/abs/1910.00935) [[Video]](https://www.youtube.com/watch?v=Z1xvAZve9aE) [[BibTex]](https://raw.githubusercontent.com/yuanming-hu/taichi/master/misc/difftaichi_bibtex.txt) [[Code]](https://github.com/yuanming-hu/difftaichi)
