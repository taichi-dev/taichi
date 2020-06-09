<div align="center">
  <img width="500px" src="https://github.com/taichi-dev/taichi/raw/master/misc/logo.png">
  <h3> <a href="https://taichi.readthedocs.io/en/stable/hello.html"> Tutorial </a> | <a href="https://github.com/taichi-dev/taichi/tree/master/examples"> Examples </a> | <a href="https://taichi.readthedocs.io/en/stable/contributor_guide.html"> Contributor Guidelines </a> | <a href="https://forum.taichi.graphics/"> Forum </a> </h3>
  <h3> <a href="https://taichi.readthedocs.io/en/stable/"> Documentation </a> | <a href="https://taichi.readthedocs.io/zh_CN/latest/"> 简体中文文档 </a> </h3>
</div>

[![Travis CI Status](https://img.shields.io/travis/taichi-dev/taichi?logo=Travis&branch=master)](https://travis-ci.com/taichi-dev/taichi)
[![AppVeyor Status](https://img.shields.io/appveyor/build/yuanming-hu/taichi?logo=AppVeyor)](https://ci.appveyor.com/project/yuanming-hu/taichi/branch/master)
[![Python Codecov Status](https://img.shields.io/codecov/c/github/taichi-dev/taichi?label=Python%20Coverage&logo=codecov)](https://codecov.io/gh/taichi-dev/taichi/src/master)
[![Latest Release](https://img.shields.io/github/v/release/taichi-dev/taichi?color=blue)](https://github.com/taichi-dev/taichi/releases/latest)

## Overview

**Taichi** (太极) is a programming language designed for *high-performance computer graphics*. It is deeply embedded in **Python**, and its **just-in-time compiler** offloads compute-intensive tasks to multi-core CPUs and massively parallel GPUs.

<a href="https://github.com/taichi-dev/taichi/blob/master/examples/fractal.py#L1-L31"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/taichi/fractal_code.png" height="293px"></a>  <img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/fractal_small.gif" height="293px">


Advanced features of Taichi include [spatially sparse computing](https://taichi.readthedocs.io/en/latest/sparse.html) and [differentiable programming](https://taichi.readthedocs.io/en/latest/differentiable_programming.html) [[examples]](https://github.com/yuanming-hu/difftaichi).

## Gallery

<a href="https://github.com/taichi-dev/taichi/blob/master/examples/mpm128.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/mpm128.gif" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/stable_fluid.py"> <img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/stable_fluids.gif" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/sdf_renderer.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/sdf_renderer.jpg" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/taichi_sparse.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/sparse_grids.gif" height="192px"></a>

<a href="https://github.com/taichi-dev/taichi/blob/master/examples/mpm_lagrangian_forces.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/lagrangian.gif" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/pbf.gif" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/game_of_life.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/game_of_life.gif" height="192px"></a> <a href="https://github.com/taichi-dev/taichi/blob/master/examples/euler.py"><img src="https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/euler.gif" height="192px"></a>

## Installation [![Downloads](https://pepy.tech/badge/taichi/month)](https://pepy.tech/project/taichi/month)

```bash
python3 -m pip install taichi
```

**Supported OS**: Windows, Linux, Mac OS X; **Python**: 3.6/3.7/3.8 (64-bit only); **Backends**: x64 CPUs, CUDA, Apple Metal, OpenGL Compute Shaders.

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

## Links

- [Taichi Forum](https://forum.taichi.graphics): Powered by Discourage, Taichi offical forum for discussing issues and sharing ideas.
- [Taichi Simplified Chinese Documentation](https://github.com/taichi-dev/taichi-docs-zh-cn): Translated by Taichi community, welcome contribution!
- [Taichi Conference](https://github.com/taichi-dev/taichicon): Taichi Online Virtual Conference available in both Chinese and English, approximately per-month.
- [GAMES 201 Lectures](https://github.com/taichi-dev/games201): (Chinese) A hands-on tutorial on advanced physics engines based on Taichi, per-week.

---

- [Taichi Elements](https://github.com/taichi-dev/taichi_elements): A High-Performance Multi-Material Continuum Physics Engine based on Taichi (work in progress).
- [LBM Taichi](https://github.com/hietwll/LBM_Taichi): Fluid solver based on Lattice Boltzmann method implemented by Taichi programming language, by [Zhuo Wang (hietwll)](https://github.com/hietwll).
- [Shadertoy reproduced by Taichi](https://github.com/Phonicavi/Shadertoy-taichi): Some prevalent shadertoys implemented in Taichi, by [QIU Feng (Phonicavi)](https://github.com/Phonicavi).
- [DiffTaichi](https://github.com/yuanming-hu/difftaichi): 10 differentiable physical simulators built with Taichi differentiable programming, by [Yuanming Hu (yuanming-hu)](https://github.com/yuanming-hu).
- [Taichi GLSL](https://github.com/yuanming-hu/difftaichi): Manipulate Taichi with GLSL-alike helper functions.

## Developers

The Taichi project was created by [Yuanming Hu (yuanming-hu)](https://github.com/yuanming-hu). Significant contributions are made by:
 - [Ye Kuang (k-ye)](https://github.com/k-ye) (Apple Metal backend)
 - [彭于斌 (archibate)](https://github.com/archibate) (OpenGL Compute Shader backend)
 - [Mingkuan Xu (xumingkuan)](https://github.com/xumingkuan) (IR optimization & standardization)

[Kenneth Lozes (KLozes)](https://github.com/KLozes) and [Yu Fang (squarefk)](https://github.com/squarefk) have also made notable contributions.

[[List of all contributors to Taichi]](https://github.com/taichi-dev/taichi/graphs/contributors)

-------------------------------

The Simplified Chinese documentation (简体中文文档) was created by [Ark (StephenArk30)](https://github.com/StephenArk30). Significant contributions are made by:
  - [彭于斌 (archibate)](https://github.com/archibate)
  - [Danni Li (isdanni)](https://github.com/isdanni)
  - [Chengchen Wang (rexwangcc)](https://github.com/rexwangcc)
  - [万健洲 (ArkhamWJZ)](https://github.com/ArkhamWJZ)

[[List of all contributors to the Simplified Chinese documentation of Taichi]](https://github.com/taichi-dev/taichi-docs-zh-cn/graphs/contributors)

-------------------------------

We welcome feedback and comments. If you would like to contribute to Taichi, please check out our [Contributor Guidelines](https://taichi.readthedocs.io/en/latest/contributor_guide.html).

If you use Taichi in your research, please cite our papers:

- [**(SIGGRAPH Asia 2019) Taichi: High-Performance Computation on Sparse Data Structures**](http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/taichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/taichi)
- [**(ICLR 2020) DiffTaichi: Differentiable Programming for Physical Simulation**](https://arxiv.org/abs/1910.00935) [[Video]](https://www.youtube.com/watch?v=Z1xvAZve9aE) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/difftaichi_bibtex.txt) [[Code]](https://github.com/yuanming-hu/difftaichi)
