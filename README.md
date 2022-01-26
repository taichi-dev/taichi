
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

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Your first Taichi program](#your-first-taichi-program)
- [Documentation](#documentation)
- [Contacts](#contacts)
- [Contributing](#contributing)
- [Resources](#resources)
  - [Demos](#demos)
  - [Lectures & talks](#lectures--talks)

**Taichi (太极)** is an open source, imperative, parallel programming language for high-performance numerical computation. It is embedded in Python, and uses just-in-time (JIT) compiler frameworks (e.g. LLVM) to offload compute-intensive Python code to native GPU or CPU instructions.

Advantages of using Taichi are:

- Built around Python: Taichi shares the same syntax with Python, allowing you to write algorithms with minimal language barrier. It's also well integrated with the Python ecosystem, such as NumPy and PyTorch.
- Flexibility: Taichi provides a set of generic data containers, namely *SNode* (/ˈsnoʊd/). SNode is an effective mechanism for composing hierarchical, multi-dimensional fields, unlocking many use patterns in numerical simulation (e.g. [spatially sparse computing](https://docs.taichi.graphics/lang/articles/advanced/sparse)).
- Performance: Through the `@ti.kernel` decorator, Taichi's JIT compiler automatically parallelizes your Python functions into efficient GPU or CPU machine code.
- Portability: Write your code once and run it everywhere. Currently, Taichi supports most of the mainstream GPU APIs, such as CUDA and Vulkan.
- ... and many more features! A cross-platform, Vulkan-based 3D visualizer, [differentiable programming](https://docs.taichi.graphics/lang/articles/advanced/differentiable_programming),  [quantized computation](https://github.com/taichi-dev/quantaichi) (experimental), etc.

# Getting Started

## Installation

You can easily install Taichi with Python's package installer `pip`:

```bash
pip install taichi
```

If you want to try out the latest features, we also provide a nightly package:

```bash
pip install -i https://test.pypi.org/simple/ taichi-nightly
```

**Supported environments**

<!--TODO: Precise OS versions-->
- Operating systems
  - Windows<sup>[1](#win-note)</sup>
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

<a name="win-note">1</a>. On Windows, please install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) first.

## Your first Taichi program

Here's how you can program a 2D fractal in Taichi:

```py
# python/taichi/examples/simulation/fractal.py

import taichi as ti

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2,
                      z[1] * z[0] * 2])

@ti.kernel
def paint(t: float):
    for i, j in pixels:
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
    while z.norm() < 20 and iterations < 50:
        z = complex_sqr(z) + c
        iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

gui = ti.GUI("Julia Set", res=(n * 2, n))

for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
```

If Taichi is properly installed, you should get the animation below 🎉:

<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/fractal.py#L1-L31"> </a><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal_small.gif" height="270px">


# Documentation

The Taichi documentations are available at:

- [Compelete documentation](https://docs.taichi.graphics/)
- [API reference](https://api-docs.taichi.graphics/)

*Due to the rapid development, both sites are still under construction. We also plan to merge them into a single site in the future.*

# Contacts

We use these channels to report bugs, discuss design, show off demos and send announcements on a daily basis:

- [GitHub Issues](https://github.com/taichi-dev/taichi/issues)
- [GitHub Discussions](https://github.com/taichi-dev/taichi/discussions)
- [Twitter](https://twitter.com/taichigraphics)
- [Taichi 中文论坛](https://forum.taichi.graphics/)
- Slack & Wechat groups: please send us a message at <a href = "mailto:contact@taichi.graphics">contact@taichi.graphics</a> first, thanks!

If you find any security problem, please do not hesitate to disclose it to <a href = "mailto:security@taichi.graphics?subject = Taichi Security Problem">security@taichi.graphics</a>.

# Contributing

If you would like to contribute to Taichi, please check out the [Contribution Guidelines](CONTRIBUTING.md).

A huge thanks to all of our amazing contributors!

<a href="https://github.com/taichi-dev/taichi/graphs/contributors"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/contributors_taichi-dev_taichi_12.png" width="800px"></a>

*Contributor avatars are randomly shuffled.*

# Resources

## Demos

- [Taichi examples](https://github.com/taichi-dev/taichi/tree/master/python/taichi/examples)
- [Advanced Taichi examples](https://github.com/taichi-dev/advanced_examples)
- [DiffTaichi](https://github.com/taichi-dev/difftaichi)
- [Taichi elements](https://github.com/taichi-dev/taichi_elements)
- [Taichi houdini](https://github.com/taichi-dev/taichi_houdini)
- [More...](misc/links.md)

<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm128.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/mpm128.gif" height="192px"></a>
<a href="https://github.com/taichi-dev/quantaichi"> <img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/smoke_3d.gif" height="192px"></a>
<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/rendering/sdf_renderer.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/sdf_renderer.jpg" height="192px"></a>
<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/euler.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/euler.gif" height="192px"></a>

<a href="https://github.com/taichi-dev/quantaichi"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/elastic_letters.gif" height="213px"></a>
<a href="https://github.com/taichi-dev/quantaichi"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fluid_with_bunnies.gif" height="213px"></a>

## Lectures & talks
- **SIGGRAPH 2020 course on Taichi basics**: [YouTube](https://youtu.be/Y0-76n3aZFA), [Bilibili](https://www.bilibili.com/video/BV1kA411n7jk/), [slides (pdf)](https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf).
- Chinagraph 2020 用太极编写物理引擎: [哔哩哔哩](https://www.bilibili.com/video/BV1gA411j7H5)
- GAMES 201 高级物理引擎实战指南2020: [课件](https://github.com/taichi-dev/games201)
- 太极图形课第一季：[课件](https://github.com/taichiCourse01)
- [TaichiCon](https://github.com/taichi-dev/taichicon): Taichi developer conferences
- More to come...

---

If you use Taichi in your research, please cite related papers:

- [**(SIGGRAPH Asia 2019) Taichi: High-Performance Computation on Sparse Data Structures**](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/taichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/taichi)
- [**(ICLR 2020) DiffTaichi: Differentiable Programming for Physical Simulation**](https://arxiv.org/abs/1910.00935) [[Video]](https://www.youtube.com/watch?v=Z1xvAZve9aE) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/difftaichi_bibtex.txt) [[Code]](https://github.com/yuanming-hu/difftaichi)
- [**(SIGGRAPH 2021) QuanTaichi: A Compiler for Quantized Simulations**](https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf) [[Video]](https://www.youtube.com/watch?v=0jdrAQOxJlY) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/quantaichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/quantaichi)
