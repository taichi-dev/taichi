<div align="center">
  <img width="500px" src="https://github.com/taichi-dev/taichi/raw/master/misc/logo.png">
   <h3> <a href="https://docs.taichi.graphics/"> Tutorial </a> | <a href="https://github.com/taichi-dev/taichi/tree/master/python/taichi/examples"> Examples </a> | <a href="https://forum.taichi.graphics/"> Forum </a><!-- | <a href="http://hub.taichi.graphics/"> Playground </a> --></h3>
  <h3> <a href="https://docs.taichi.graphics/"> Documentation </a> | <a href="https://docs.taichi.graphics/zh-Hans/"> 简体中文文档 </a> | <a href="https://docs.taichi.graphics/lang/articles/contribution/contributor_guide"> Contributor Guidelines </a> </h3>
</div>

[![Postsubmit Checks](https://github.com/taichi-dev/taichi/actions/workflows/release.yml/badge.svg)](https://github.com/taichi-dev/taichi/actions/workflows/release.yml)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/taichidev/taichi?label=Docker%20Image&logo=docker)](https://hub.docker.com/r/taichidev/taichi)
[![Python Codecov Status](https://img.shields.io/codecov/c/github/taichi-dev/taichi?label=Python%20Coverage&logo=codecov)](https://codecov.io/gh/taichi-dev/taichi/src/master)
[![Latest Release](https://img.shields.io/github/v/release/taichi-dev/taichi?color=blue&label=Latest%20Release)](https://github.com/taichi-dev/taichi/releases/latest)
[![Netlify Status](https://api.netlify.com/api/v1/badges/6825e411-c5f7-4148-ab43-023663f41b6a/deploy-status)](https://app.netlify.com/sites/docs-taichi-graphics/deploys)

## Overview

**Taichi** (太极) is a parallel programming language for high-performance numerical computations. It is embedded in **Python**, and its **just-in-time compiler** offloads compute-intensive tasks to multi-core CPUs and massively parallel GPUs.

<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/fractal.py#L1-L31"> <img src="https://github.com/taichi-dev/public_files/raw/master/taichi/fractal_code.png" height="270px"></a>  <img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal_small.gif" height="270px">

Advanced features of Taichi include [spatially sparse computing](https://docs.taichi.graphics/lang/articles/advanced/sparse), [differentiable programming](https://docs.taichi.graphics/lang/articles/advanced/differentiable_programming) [[examples]](https://github.com/yuanming-hu/difftaichi), and [quantized computation](https://github.com/taichi-dev/quantaichi).

**Please check out our SIGGRAPH 2020 course on Taichi basics:** [YouTube](https://youtu.be/Y0-76n3aZFA), [Bilibili](https://www.bilibili.com/video/BV1kA411n7jk/), [slides (pdf)](https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf).

**中文视频教程:** [[哔哩哔哩]](https://www.bilibili.com/video/BV1gA411j7H5), [[幻灯片]](https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf)

## Examples ([More...](misc/examples.md))

<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm128.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/mpm128.gif" height="192px"></a>
<a href="https://github.com/taichi-dev/quantaichi"> <img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/smoke_3d.gif" height="192px"></a>
<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/rendering/sdf_renderer.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/sdf_renderer.jpg" height="192px"></a>
<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/euler.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/euler.gif" height="192px"></a>

<a href="https://github.com/taichi-dev/quantaichi"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/elastic_letters.gif" height="213px"></a>
<a href="https://github.com/taichi-dev/quantaichi"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fluid_with_bunnies.gif" height="213px"></a>

## Installation [![Downloads](https://pepy.tech/badge/taichi)](https://pepy.tech/project/taichi)

### Official releases

```bash
python3 -m pip install taichi
```

**Supported OS**: Windows, Linux, Mac OS X; **Python**: 3.6-3.9 (64-bit only); **Backends**: x64 CPUs, CUDA, Apple Metal, Vulkan, OpenGL Compute Shaders.

### Nightly releases

Nightly releases with the `master` branch of `taichi` are also available:

```bash
pip install -i https://test.pypi.org/simple/ taichi-nightly
```

Note nightly releases are bleeding edge versions and thus _may_ and _will_ contain bugs. Those releases are primarily aimed for alpha feature testing.
If you need a stable version, please refer to the official release above.

### Building from source

Please build from source for other configurations (e.g., your CPU is ARM, or you want to try out our experimental C backend).

**Note:**
 - The PyPI package supports x64 CPU, CUDA 10/11, Metal, Vulkan and OpenGL Compute Shader backends.
 - On Windows, please install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) if you haven't.
 - [[All releases]](https://github.com/taichi-dev/taichi/releases)

## Contributing

We'd love to hear your comments or any of your feedback! If you would like to contribute to Taichi, please check out the [Contribution Guidelines](CONTRIBUTING.md) first.

## Contributors

<a href="https://github.com/taichi-dev/taichi/graphs/contributors"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/contributors_taichi-dev_taichi_12.png" width="800px"></a>

*Note: contributor avatars above are randomly shuffled.*

-------------------------------

If you use Taichi in your research, please cite related papers:

- [**(SIGGRAPH Asia 2019) Taichi: High-Performance Computation on Sparse Data Structures**](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/taichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/taichi)
- [**(ICLR 2020) DiffTaichi: Differentiable Programming for Physical Simulation**](https://arxiv.org/abs/1910.00935) [[Video]](https://www.youtube.com/watch?v=Z1xvAZve9aE) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/difftaichi_bibtex.txt) [[Code]](https://github.com/yuanming-hu/difftaichi)
- [**(SIGGRAPH 2021) QuanTaichi: A Compiler for Quantized Simulations**](https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf) [[Video]](https://www.youtube.com/watch?v=0jdrAQOxJlY) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/quantaichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/quantaichi)

## Links
- [TaichiCon](https://github.com/taichi-dev/taichicon): Taichi developer conferences.
- [GAMES 201 Lectures](https://github.com/taichi-dev/games201): (Chinese) A hands-on course on building advanced physics engines, based on Taichi.
- [TaichiZoo](https://zoo.taichi.graphics): Running Taichi code in your browser <sup>[1](#zoo-disclaimer)</sup>.
- [加入太极图形](https://app.mokahr.com/apply/taichi/41024#/).
- [太极图形课](https://github.com/taichiCourse01).
- [More...](misc/links.md)

## Security

Please disclose security issues responsibly to <a href = "mailto:security@taichi.graphics?subject = Security Problems">security@taichi.graphics</a>.

---

<a name="zoo-disclaimer">1</a>. TaichiZoo is still in its Beta version. If you've encountered any issue, please do not hesitate to [file a bug](https://github.com/taichi-dev/taichi-zoo-issue-tracker/issues/new/choose).
