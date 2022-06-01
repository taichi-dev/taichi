<div align="center">
  <img width="500px" src="https://github.com/taichi-dev/taichi/raw/master/misc/logo.png"/>
</div>

---

[![Latest Release](https://img.shields.io/github/v/release/taichi-dev/taichi?color=blue&label=Latest%20Release)](https://github.com/taichi-dev/taichi/releases/latest)
[![downloads](https://pepy.tech/badge/taichi)](https://pepy.tech/project/taichi)
[![CI](https://github.com/taichi-dev/taichi/actions/workflows/testing.yml/badge.svg)](https://github.com/taichi-dev/taichi/actions/workflows/testing.yml)
[![Nightly Release](https://github.com/taichi-dev/taichi/actions/workflows/release.yml/badge.svg)](https://github.com/taichi-dev/taichi/actions/workflows/release.yml)
[![Python Codecov Status](https://img.shields.io/codecov/c/github/taichi-dev/taichi?label=Python%20Coverage&logo=codecov)](https://codecov.io/gh/taichi-dev/taichi/src/master)

```py
import taichi as ti
```

## What is Taichi Lang?

Taichi Lang is an open-source, imperative, parallel programming language for high-performance numerical computation. It is embedded in Python and uses just-in-time (JIT) compiler frameworks, for example LLVM, to offload the compute-intensive Python code to the native GPU or CPU instructions.

<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/fractal.py#L1-L31"> <img src="https://github.com/taichi-dev/public_files/raw/master/taichi/fractal_code.png" height="270px"></a>  <img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal_small.gif" height="270px">

The language has broad applications spanning real-time physical simulation, numberical computation, augmented reality, artificial intelligence, vision and robotics, visual effects in films and games, general-purpose computing, and much more.

<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm128.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/mpm128.gif" height="192px"></a>
<a href="https://github.com/taichi-dev/quantaichi"> <img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/smoke_3d.gif" height="192px"></a>
<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/rendering/sdf_renderer.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/sdf_renderer.jpg" height="192px"></a>
<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/euler.py"><img src="https://github.com/taichi-dev/public_files/raw/master/taichi/euler.gif" height="192px"></a>

<a href="https://github.com/taichi-dev/quantaichi"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/elastic_letters.gif" height="213px"></a>
<a href="https://github.com/taichi-dev/quantaichi"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fluid_with_bunnies.gif" height="213px"></a>

[...More](#demos)

## Why Taichi Lang?

- Built around Python: Taichi Lang shares almost the same syntax with Python, allowing you to write algorithms with minimal language barrier. It is also well integrated into the Python ecosystem, including NumPy and PyTorch.
- Flexibility: Taichi Lang provides a set of generic data containers known as *SNode* (/ˈsnoʊd/), an effective mechanism for composing hierarchical, multi-dimensional fields. This can cover many use patterns in numerical simulation (e.g. [spatially sparse computing](https://docs.taichi-lang.org/docs/sparse)).
- Performance: With the `@ti.kernel` decorator, Taichi Lang's JIT compiler automatically compiles your Python functions into efficient GPU or CPU machine code for parallel execution.
- Portability: Write your code once and run it everywhere. Currently, Taichi Lang supports most mainstream GPU APIs, such as CUDA and Vulkan.
- ... and many more features! A cross-platform, Vulkan-based 3D visualizer, [differentiable programming](https://docs.taichi-lang.org/docs/differentiable_programming),  [quantized computation](https://github.com/taichi-dev/quantaichi) (experimental), etc.

## Getting Started

### Installation

<details>
  <summary>Prerequisites</summary>

<!--TODO: Precise OS versions-->

- Operating systems
  - Windows
  - Linux
  - macOS
- Python: 3.6 ~ 3.10 (64-bit only)
- Compute backends
  - x64/ARM CPUs
  - CUDA
  - Vulkan
  - OpenGL (4.3+)
  - Apple Metal
  - WebAssembly (experiemental)
 </details>

Use Python's package installer **pip** to install Taichi Lang:

```bash
pip install --upgrade taichi
```

*We also provide a nightly package. Note that nighly packages may crash because they are not fully tested.  We cannot guarantee their validity, and you are at your own risk trying out our latest, untested features. The nightly packages can be installed from our self-hosted PyPI (Using self-hosted PyPI allows us to provide more frequent releases over a longer period of time)*

```bash
pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly
```

### Run your "Hello, world!"

Here is how you can program a 2D fractal in Taichi:

```py
# python/taichi/examples/simulation/fractal.py

import taichi as ti

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
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
    gui.show()
```

*If Taichi Lang is properly installed, you should get the animation below 🎉:*

<a href="https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/fractal.py#L1-L31"> </a><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal_small.gif" height="270px">

See [Get started](https://docs.taichi-lang.org) for more information.

### Build from source

If you wish to try our our experimental features or build Taichi Lang for your own environments, see [Developer installation](https://docs.taichi-lang.org/docs/dev_install).

## Documentation

- [Technical documents](https://docs.taichi-lang.org/)
- [API Reference](https://docs.taichi-lang.org/api/)
- [Blog](https://docs.taichi-lang.org/blog)

## Contributing

Kudos to all of our amazing contributors! Taichi Lang thrives through open-source. In that spirit, we welcome all kinds of contributions from the community. If you would like to participate, check out the [Contribution Guidelines](CONTRIBUTING.md) first.

<a href="https://github.com/taichi-dev/taichi/graphs/contributors"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/contributors_taichi-dev_taichi_12.png" width="800px"></a>

*Contributor avatars are randomly shuffled.*

## License

Taichi Lang is distributed under the terms of Apache License (Version 2.0).

See [Apache License](https://github.com/taichi-dev/taichi/blob/master/LICENSE) for details.

## Community

### Event

Voxel Challenge 2022 is open for [submissions](https://github.com/taichi-dev/voxel-challenge/issues/11) until 18th May. Find out more [here](https://github.com/taichi-dev/community/tree/main/events/voxel-challenge).

### Join our discussions

- [Slack Channel](https://join.slack.com/t/taichicommunity/shared_invite/zt-14ic8j6no-Fd~wKNpfskXLfqDr58Tddg)
- [GitHub Discussions](https://github.com/taichi-dev/taichi/discussions)
- [太极编程语言中文论坛](https://forum.taichi.graphics/)

### Report an issue

- If you spot an technical or documentation issue, file an issue at [GitHub Issues](https://github.com/taichi-dev/taichi/issues)
- If you spot any security issue, mail directly to <a href = "mailto:security@taichi.graphics?subject = Taichi Security Problem">security@taichi.graphics</a>.

### Contact us

- [Slack](https://taichicommunity.slack.com/join/shared_invite/zt-14ic8j6no-Fd~wKNpfskXLfqDr58Tddg#/shared-invite/email)
- WeChat: Drop us a message at <a href = "mailto:community@taichi.graphics">community@taichi.graphics</a> first, and we'll follow up.

## Reference

### Demos

- [Taichi Lang examples](https://github.com/taichi-dev/taichi/tree/master/python/taichi/examples)
- [Advanced Taichi Lang examples](https://github.com/taichi-dev/advanced_examples)
- [DiffTaichi](https://github.com/taichi-dev/difftaichi)
- [Taichi elements](https://github.com/taichi-dev/taichi_elements)
- [Taichi Houdini](https://github.com/taichi-dev/taichi_houdini)
- [More...](misc/links.md)

### Lectures & talks

- SIGGRAPH 2020 course on Taichi basics: [YouTube](https://youtu.be/Y0-76n3aZFA), [Bilibili](https://www.bilibili.com/video/BV1kA411n7jk/), [slides (pdf)](https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf).
- Chinagraph 2020 用太极编写物理引擎: [哔哩哔哩](https://www.bilibili.com/video/BV1gA411j7H5)
- GAMES 201 高级物理引擎实战指南 2020: [课件](https://github.com/taichi-dev/games201)
- 太极图形课第一季：[课件](https://github.com/taichiCourse01)
- [TaichiCon](https://github.com/taichi-dev/taichicon): Taichi Developer Conferences
- More to come...

### Citations

If you use Taichi Lang in your research, please cite the corresponding papers:

- [**(SIGGRAPH Asia 2019) Taichi: High-Performance Computation on Sparse Data Structures**](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/taichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/taichi)
- [**(ICLR 2020) DiffTaichi: Differentiable Programming for Physical Simulation**](https://arxiv.org/abs/1910.00935) [[Video]](https://www.youtube.com/watch?v=Z1xvAZve9aE) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/difftaichi_bibtex.txt) [[Code]](https://github.com/yuanming-hu/difftaichi)
- [**(SIGGRAPH 2021) QuanTaichi: A Compiler for Quantized Simulations**](https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf) [[Video]](https://www.youtube.com/watch?v=0jdrAQOxJlY) [[BibTex]](https://raw.githubusercontent.com/taichi-dev/taichi/master/misc/quantaichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/quantaichi)
