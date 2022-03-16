<div align="center">
  <img width="500px" src="https://github.com/taichi-dev/taichi/raw/master/misc/logo.png"/>
</div>


---

[![Latest Release](https://img.shields.io/github/v/release/taichi-dev/taichi?color=blue&label=Latest%20Release)](https://github.com/taichi-dev/taichi/releases/latest)
[![downloads](https://pepy.tech/badge/taichi)](https://pepy.tech/project/taichi)
[![CI](https://github.com/taichi-dev/taichi/actions/workflows/testing.yml/badge.svg)](https://github.com/taichi-dev/taichi/actions/workflows/postsubmit.yml)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/taichidev/taichi?label=Docker%20Image&logo=docker)](https://hub.docker.com/r/taichidev/taichi)
[![Python Codecov Status](https://img.shields.io/codecov/c/github/taichi-dev/taichi?label=Python%20Coverage&logo=codecov)](https://codecov.io/gh/taichi-dev/taichi/src/master)

```py
import taichi as ti
```

# What is Taichi?

Taichi is an open-source, imperative, parallel programming language for high-performance numerical computation. It is embedded in Python and uses just-in-time (JIT) compiler frameworks, for example LLVM, to offload the compute-intensive Python code to the native GPU or CPU instructions.

<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/fractal.py#L1-L31"> <img src="https://github.com/taichi-dev/public_files/raw/master/taichi/fractal_code.png" height="270px"></a>  <img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal_small.gif" height="270px">

The language has broad applications spanning real-time physical simulation, numberical computation, augmented reality, artificial intelligence, vision and robotics, visual effects in films and games, general-purpose computing, and much more.

<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm128.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/mpm128.gif" height="192px"></a>
<a href="https://github.com/taichi-dev/quantaichi"> <img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/smoke_3d.gif" height="192px"></a>
<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/rendering/sdf_renderer.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/sdf_renderer.jpg" height="192px"></a>
<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/euler.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/euler.gif" height="192px"></a>

<a href="https://github.com/taichi-dev/quantaichi"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/elastic_letters.gif" height="213px"></a>
<a href="https://github.com/taichi-dev/quantaichi"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fluid_with_bunnies.gif" height="213px"></a>

# Why Taichi?

- Fast development
   - Simple syntax is elegance: No barrier to entry for Python users.
   - Naturally integrated into the Python ecosystem, including NumPy and PyTorch.
   - Automatic parallelization and differentiation spare you the implementation efforts.
   - Develop fancy computer graphics programs in less than 99 lines of code!
- High performance
   - Born to harness parallelism in GPUs and multi-core CPUs 
   - Compiled Python just-in-time to binary executable kernels.
   - Spatially sparse data structures: No wasted computation in empty space.
   - Quantized computation optimizes performance on mobile devices.
- Universal deployment
   - Supports multiple backends including x64 and ARM CPUs, CUDA, Vulkan, Metal, and OpenGL Compute Shaders.
   - Ahead-of-time compilation enables deployment on platforms without Python, including PCs, mobile devices, and even web browsers.

And there are a lot more great features for you to discover: SNode (/ˈsnoʊd/), an effective mechanism for composing hierarchical, multi-dimensional fields, a cross-platform, Vulkan-based 3D visualizer, [spatially sparse computation](https://docs.taichi.graphics/lang/articles/advanced/sparse), [differentiable programming](https://docs.taichi.graphics/lang/articles/advanced/differentiable_programming),  [quantized computation](https://github.com/taichi-dev/quantaichi) (experimental)...

# Getting Started

### Prerequisites

<!--TODO: Precise OS versions-->

- Operating systems
  - Windows
  - Linux
  - macOS
- Python: 3.6 ~ 3.9 (64-bit only)
- Compute backends
  - x64/ARM CPUs
  - CUDA
  - Vulkan
  - OpenGL (4.3+)
  - Apple Metal
  - WebAssembly (experiemental)

### Installation

Use Python's package installer **pip** to install Taichi:

```bash
pip install taichi
```

> *We also provide a nightly package. Note that nighly packages may crash because they are not fully tested.  We cannot guarantee their validity, and you are at your own risk trying out our latest, untest features. *

```bash
pip install -i https://test.pypi.org/simple/ taichi-nightly
```

### Run your "Hello, world!"

See [Hello,world!](https://docs.taichi.graphics/#hello-world).

### Build from source

If you wish to try our our experimental features or build Taichi for your own environments, see [Developer installation](https://docs.taichi.graphics/lang/articles/contribution/dev_install).

# Reference

### Documentation

- [Release information](https://github.com/taichi-dev/taichi/releases)
- [Technical documents](https://docs.taichi.graphics/)
- [API Reference](https://docs.taichi.graphics/api/)
- [Blog](https://docs.taichi.graphics/blog)

### Demos

- [Taichi examples](https://github.com/taichi-dev/taichi/tree/master/python/taichi/examples)
- [Advanced Taichi examples](https://github.com/taichi-dev/advanced_examples)
- [DiffTaichi](https://github.com/taichi-dev/difftaichi)
- [Taichi elements](https://github.com/taichi-dev/taichi_elements)
- [Taichi houdini](https://github.com/taichi-dev/taichi_houdini)
- [More...](misc/links.md)

### Lectures & talks

- SIGGRAPH 2020 course on Taichi basics: [YouTube](https://youtu.be/Y0-76n3aZFA), [Bilibili](https://www.bilibili.com/video/BV1kA411n7jk/), [slides (pdf)](https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf).
- Chinagraph 2020 用太极编写物理引擎: [哔哩哔哩](https://www.bilibili.com/video/BV1gA411j7H5)
- GAMES 201 高级物理引擎实战指南 2020: [课件](https://github.com/taichi-dev/games201)
- 太极图形课第一季：[课件](https://github.com/taichiCourse01)
- [TaichiCon](https://github.com/taichi-dev/taichicon): Taichi Developer Conferences
- More to come...

### Citations

If you use Taichi in your research, please cite the corresponding papers:

- [(SIGGRAPH Asia 2019) Taichi: High-Performance Computation on Sparse Data Structures](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo)|[[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/taichi_bibtex.txt)|[[Code]](https://github.com/taichi-dev/taichi)
- [(ICLR 2020) DiffTaichi: Differentiable Programming for Physical Simulation](https://arxiv.org/abs/1910.00935) [[Video]](https://www.youtube.com/watch?v=Z1xvAZve9aE)|[[BibTex]|[[Code]](https://github.com/yuanming-hu/difftaichi)
- [(SIGGRAPH 2021) QuanTaichi: A Compiler for Quantized Simulations](https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf) [[Video]|[[BibTex]|[[Code]](https://github.com/taichi-dev/quantaichi)

# Contributing

Kudos to all of our amazing contributors! Taichi thrives through open-source. In that spirit, we welcome all kinds of contributions from the community. If you would like to participate, check out the [Contribution Guidelines](CONTRIBUTING.md) first.

<a href="https://github.com/taichi-dev/taichi/graphs/contributors"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/contributors_taichi-dev_taichi_12.png" width="800px"></a>

*Contributor avatars are randomly shuffled.*

# Community

### Join our discussions

- [GitHub Discussions](https://github.com/taichi-dev/taichi/discussions)
- [Taichi 中文论坛](https://forum.taichi.graphics/)

### Report an issue

- If you spot an technical or documentation issue, file an issue at [GitHub Issues](https://github.com/taichi-dev/taichi/issues)
- If you spot any security issue, mail directly to <a href = "mailto:security@taichi.graphics?subject = Taichi Security Problem">security@taichi.graphics</a>.

### Contact us

You can also join our community from Slack or WeChat. Drop us a message at <a href = "mailto:contact@taichi.graphics">contact@taichi.graphics</a> first, and we'll follow up.

### Follow us on social media

- [Twitter](https://twitter.com/TaichiGraphics)
- [LinkedIn](https://www.linkedin.com/company/taichi-graphics/)
- [YouTube](https://www.youtube.com/channel/UCu-k1Wglo9Ll_o2j5Bxl4cw/featured)

### Join us

The Taichi programming language is powered by Taichi Graphics. The team is rapidly expanding. Find us from the following channels to get a taste of our *vibe*. If you feel you can fit in, welcome to join us!:-)

- [Taichi Graphics](https://taichi.graphics/)
- [Careers at Taichi](https://taichi.graphics/careers/)
