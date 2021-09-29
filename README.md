<div align="center">
  <img width="500px" src="https://github.com/taichi-dev/taichi/raw/master/misc/logo.png">
   <h3> <a href="https://docs.taichi.graphics/"> Tutorial </a> | <a href="https://github.com/taichi-dev/taichi/tree/master/examples"> Examples </a> | <a href="https://forum.taichi.graphics/"> Forum </a><!-- | <a href="http://hub.taichi.graphics/"> Playground </a> --></h3>
  <h3> <a href="https://docs.taichi.graphics/"> Documentation </a> | <a href="https://docs.taichi.graphics/zh-Hans/docs/"> 简体中文文档 </a> | <a href="https://docs.taichi.graphics/docs/lang/articles/contribution/contributor_guide"> Contributor Guidelines </a> </h3>
</div>

[![AppVeyor Status](https://img.shields.io/appveyor/build/yuanming-hu/taichi?logo=AppVeyor&label=AppVeyor)](https://ci.appveyor.com/project/yuanming-hu/taichi/branch/master)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/taichidev/taichi?label=Docker%20Image&logo=docker)](https://hub.docker.com/r/taichidev/taichi)
[![Python Codecov Status](https://img.shields.io/codecov/c/github/taichi-dev/taichi?label=Python%20Coverage&logo=codecov)](https://codecov.io/gh/taichi-dev/taichi/src/master)
[![Latest Release](https://img.shields.io/github/v/release/taichi-dev/taichi?color=blue&label=Latest%20Release)](https://github.com/taichi-dev/taichi/releases/latest)

## Overview

**Taichi** (太极) is a parallel programming language for high-performance numerical computations. It is embedded in **Python**, and its **just-in-time compiler** offloads compute-intensive tasks to multi-core CPUs and massively parallel GPUs.

<a href="https://github.com/taichi-dev/taichi/blob/master/examples/simulation/fractal.py#L1-L31"> <img src="https://github.com/taichi-dev/public_files/raw/master/taichi/fractal_code.png" height="270px"></a>  <img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal_small.gif" height="270px">

Advanced features of Taichi include [spatially sparse computing](https://docs.taichi.graphics/docs/lang/articles/advanced/sparse), [differentiable programming](https://docs.taichi.graphics/docs/lang/articles/advanced/differentiable_programming) [[examples]](https://github.com/yuanming-hu/difftaichi), and [quantized computation](https://github.com/taichi-dev/quantaichi).

**Please check out our SIGGRAPH 2020 course on Taichi basics:** [YouTube](https://youtu.be/Y0-76n3aZFA), [Bilibili](https://www.bilibili.com/video/BV1kA411n7jk/), [slides (pdf)](https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf).

**中文视频教程:** [[哔哩哔哩]](https://www.bilibili.com/video/BV1gA411j7H5), [[幻灯片]](https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf)

## Examples ([More...](misc/examples.md))

<a href="https://github.com/taichi-dev/taichi/blob/master/examples/simulation/mpm128.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/mpm128.gif" height="192px"></a>
<a href="https://github.com/taichi-dev/quantaichi"> <img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/smoke_3d.gif" height="192px"></a>
<a href="https://github.com/taichi-dev/taichi/blob/master/examples/rendering/sdf_renderer.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/sdf_renderer.jpg" height="192px"></a>
<a href="https://github.com/taichi-dev/taichi/blob/master/examples/simulation/euler.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/euler.gif" height="192px"></a>

<a href="https://github.com/taichi-dev/quantaichi"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/elastic_letters.gif" height="213px"></a>
<a href="https://github.com/taichi-dev/quantaichi"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fluid_with_bunnies.gif" height="213px"></a>

## Installation [![Downloads](https://pepy.tech/badge/taichi)](https://pepy.tech/project/taichi)

```bash
python3 -m pip install taichi
```

**Supported OS**: Windows, Linux, Mac OS X; **Python**: 3.6-3.9 (64-bit only); **Backends**: x64 CPUs, CUDA, Apple Metal, OpenGL Compute Shaders.

Please build from source for other configurations (e.g., your CPU is ARM, or you want to try out our experimental C backend).

**Note:**
 - The PyPI package supports x64 CPU, CUDA 10/11, Metal, and OpenGL Compute Shader backends.
 - On Ubuntu 19.04+, please `sudo apt install libtinfo5`.
 - On Windows, please install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) if you haven't.
 - [[All releases]](https://github.com/taichi-dev/taichi/releases)

|| **Linux (CUDA)** | **OS X (10.14+)** | **Windows** | **Documentation**|
|:------|:-----|:-----|:-----|:-----|
|**Build**|[![Build Status](http://f11.csail.mit.edu:8080/job/taichi/badge/icon)](http://f11.csail.mit.edu:8080/job/taichi/)| [![Build Status](https://travis-ci.com/taichi-dev/taichi.svg?branch=master)](https://travis-ci.com/taichi-dev/taichi) | [![Build status](https://ci.appveyor.com/api/projects/status/yxm0uniin8xty4j7/branch/master?svg=true)](https://ci.appveyor.com/project/yuanming-hu/taichi/branch/master)| [![Netlify Status](https://api.netlify.com/api/v1/badges/6825e411-c5f7-4148-ab43-023663f41b6a/deploy-status)](https://app.netlify.com/sites/docs-taichi-graphics/deploys)|
|**PyPI**|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build status](https://ci.appveyor.com/api/projects/status/39ar9wa8yd49je7o?svg=true)](https://ci.appveyor.com/project/yuanming-hu/taichi-wheels-test) |

## Developer Installation

Please follow [this doc](https://docs.taichi.graphics/docs/lang/articles/contribution/dev_install) to learn how to build Taichi from source. Note that Taichi requires LLVM-10.0.0, and it is recommneded to use [our prebuilt LLVM libraries](https://docs.taichi.graphics/docs/lang/articles/contribution/dev_install#installing-dependencies) for each platform.

## Contributors

<a href="https://github.com/taichi-dev/taichi/graphs/contributors"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/contributors_taichi-dev_taichi_12.png" width="800px"></a>

*Note: contributor avatars above are randomly shuffled.*

-------------------------------

We welcome feedback and comments. If you would like to contribute to Taichi, please check out our [Contributor Guidelines](https://docs.taichi.graphics/docs/lang/articles/contribution/contributor_guide).

If you use Taichi in your research, please cite related papers:

- [**(SIGGRAPH Asia 2019) Taichi: High-Performance Computation on Sparse Data Structures**](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/taichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/taichi)
- [**(ICLR 2020) DiffTaichi: Differentiable Programming for Physical Simulation**](https://arxiv.org/abs/1910.00935) [[Video]](https://www.youtube.com/watch?v=Z1xvAZve9aE) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/difftaichi_bibtex.txt) [[Code]](https://github.com/yuanming-hu/difftaichi)
- [**(SIGGRAPH 2021) QuanTaichi: A Compiler for Quantized Simulations**](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf) [[Video]](https://www.youtube.com/watch?v=0jdrAQOxJlY) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/quantaichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/quantaichi)

## Links
- [TaichiCon](https://github.com/taichi-dev/taichicon): Taichi developer conferences.
- [GAMES 201 Lectures](https://github.com/taichi-dev/games201): (Chinese) A hands-on course on building advanced physics engines, based on Taichi.
- [TaichiZoo](https://zoo.taichi.graphics): Running Taichi code in your browser <sup>[1](#zoo-disclaimer)</sup>.
- [加入太极图形](https://app.mokahr.com/apply/taichi/41024#/).
- [太极图形课](https://github.com/taichiCourse01).
- [More...](misc/links.md)

## Security

Please disclose security issues responsibly to contact@taichi.graphics.

---

<a name="zoo-disclaimer">1</a>. TaichiZoo is still in its Beta version. If you've encountered any issue, please do not hesitate to [file a bug](https://github.com/taichi-dev/taichi-zoo-issue-tracker/issues/new/choose).
