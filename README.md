<div align="center">
  <img width="500px" src="https://github.com/yuanming-hu/taichi/raw/master/misc/logo.png">
  <h3> <a href="https://taichi.readthedocs.io/en/latest/"> Docs </a> | <a href="https://taichi.readthedocs.io/en/latest/hello.html"> Tutorial </a> | <a href="https://github.com/yuanming-hu/difftaichi"> DiffTaichi </a> | <a href="https://github.com/yuanming-hu/taichi/tree/master/examples"> Examples </a> | <a href="https://taichi.readthedocs.io/en/latest/contributor_guide.html"> Contribute </a> | <a href="https://forum.taichi.graphics/"> Forum </a> </h3>
</div>

| **Documentations** | **Chat** | Downloads |
|:-----|:-----|:----|
| [![Documentation Status](https://readthedocs.org/projects/taichi/badge/?version=latest)](http://taichi.readthedocs.io/en/latest/?badge=latest) | [![Join the chat at https://gitter.im/taichi-dev/Lobby](https://badges.gitter.im/taichi-dev/Lobby.svg)](https://gitter.im/taichi-dev/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) | [![Downloads](https://pepy.tech/badge/taichi-nightly)](https://pepy.tech/project/taichi-nightly) |

```bash
python3 -m pip install taichi
```
**Supported environments:**
 - OS: Windows, Linux, and OS X
 - Python: 3.6/3.7/3.8
 - Backends: x64 CPUs/CUDA/Apple Metal.
 - Build from source if you work in other environments (e.g., you need the OpenGL backend or your CPU is ARM).

**Note:**
 - Since April 12 2020 (v0.5.11), we release the Python package [`taichi`](https://pypi.org/project/taichi/) instead of [`taichi-nightly`](https://pypi.org/project/taichi-nightly/). This PyPI package has CPU, CUDA 10 and Metal support.
 - On Ubuntu 19.04+, please `sudo apt install libtinfo5`.
 - On Windows, please install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) if you haven't.

## [Contribution Guidelines](https://taichi.readthedocs.io/en/latest/contributor_guide.html)

|| **Linux (CUDA)** | **OS X (10.14+)** | **Windows** |
|:------|:-----|:-----|:-----|
|**Build**|[![Build Status](http://f11.csail.mit.edu:8080/job/taichi/badge/icon)](http://f11.csail.mit.edu:8080/job/taichi/)| [![Build Status](https://travis-ci.com/taichi-dev/taichi.svg?branch=master)](https://travis-ci.com/taichi-dev/taichi) | [![Build status](https://ci.appveyor.com/api/projects/status/yxm0uniin8xty4j7/branch/master?svg=true)](https://ci.appveyor.com/project/yuanming-hu/taichi/branch/master)|
|**PyPI**|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build status](https://ci.appveyor.com/api/projects/status/39ar9wa8yd49je7o?svg=true)](https://ci.appveyor.com/project/IteratorAdvance/taichi-wheels-test)|

## Updates
- (Mar 29, 2020) v0.5.10 released
   - **Language and syntax**
      - Fix `ti.static(ti.grouped(ti.ndrange(...)))` syntax checker false positive (#680) (by **Yuanming Hu**)
   - **Command line interface**
      - `ti test` now supports `-t/--threads` for specifying number of testing threads (#674) (by **Yuanming Hu**)
   - [Full log](https://github.com/taichi-dev/taichi/releases/tag/0.5.10)

- (Mar 28, 2020) v0.5.9 released
   - **CPU backends**
      - Support `bitmasked` as the leaf block structure for `1x1x1` masks (#676) (by **Yuanming Hu**)
   - **CUDA backend**
      - Support `bitmasked` as the leaf block structure for `1x1x1` masks (#676) (by **Yuanming Hu**)
   - **Documentation**
      - Updated contributor guideline (#658) (by **Yuanming Hu**)
   - **Infrastructure**
      - 6x faster compilation on CPU/CUDA backends (#673) (by **Yuanming Hu**)
   - **Language and syntax**
      - Simplify dense.bitmasked to bitmasked (#670) (by **Ye Kuang**)
      - Support break in non-parallel for statements (#583) (by **彭于斌**)
   - **Metal backend**
      - Enable `bitmasked` on Metal (#661) (by **Ye Kuang**)
      - Silence compile warnings (#650) (by **Ye Kuang**)
   - **Optimization**
      - Improved Taichi IR optimizers (#662) (#668) (by **xumingkuan**)
   - [Full log](https://github.com/taichi-dev/taichi/releases/tag/0.5.9)
- (Mar  24, 2020) v0.5.8 released. Visible/notable changes:
   - **Language features**
      - Access out-of-bound checking on CPU backends (#572) (by **xumingkuan**)
      - Testable device-side assertion failures on CPUs (#605) (by **xumingkuan**)
      - Added `Expr.copy_from` (by **Yuanming Hu**)
      - Added `ti.Vector.unit` to generate unit vectors (by **Yuanming Hu**)
      - Use `a = ti.static(a_very_long_variable)` for easy aliasing [[doc]](https://taichi.readthedocs.io/en/latest/syntax_sugars.html#aliases) (#587) (by **彭于斌** and **KLozes**)
      - Added  `ti.atomic_min`,  `ti.atomic_max`, `ti.atomic_bit_or`, `ti.atomic_bit_and`, `ti.atomic_bit_xor` (CPU and CUDA by **KLozes**, OpenGL by **彭于斌**, Metal by **Ye Kuang**)
   - **Differentiable programming**
      - Experimental support for automatically differentiating through conditional global load/stores (by **Yuanming Hu**)
   - **Bug fixes**
      - Fixed stack traceback printing on OS X (#610) (by **Yuanming Hu**)
   - **CLI**
      - `ti format` now cover all files from upstream/master to the working tree (#629) (by **Ye Kuang**)
      - `ti test` now uses `argparse` for better customizability (#601) (by **彭于斌**)
   - **OpenGL backend**
      - OpenGL Compute Shader backend will officially release very soon with v0.6! (by **彭于斌**)
   - **Metal backend**
      - Metal backend sparsity support work in progress (by **Ye Kuang**)
   - **Examples**
      - Added `examples/mgpcg.py` (#573) (by **KLozes**)
      - Added `examples/sdf_renderer.py` (by **Yuanming Hu**)
      - Added `examples/mgpcg_advanced.py` (#573) (by **Yuanming Hu**)
   - [Full log](https://github.com/taichi-dev/taichi/releases/tag/0.5.8)
- (Mar   4, 2020) v0.5.7 released
   - **Deprecated `ti.classfunc`. Please simply use `ti.func`, even if you are decorating a class member function**
   - Upgrade spdlog from 0.14.0 to 1.5.0 with git submodule (#565) (by **Mingkuan Xu [xumingkuan]**)
   - Metal backend refactored (by **Ye Kuang [k-ye]**)
   - Fixed infinitely looping signal handlers
   - Fixed `ti test` on release mode
   - Doc updated
- [Full history](changelog.md)


## Related papers
- [**(ICLR 2020) Differentiable Programming for Physical Simulation**](https://arxiv.org/abs/1910.00935) [[Video]](https://www.youtube.com/watch?v=Z1xvAZve9aE) [[BibTex]](https://raw.githubusercontent.com/yuanming-hu/taichi/master/misc/difftaichi_bibtex.txt) [[Code]](https://github.com/yuanming-hu/difftaichi)
  - by *Yuanming Hu, Luke Anderson, Tzu-Mao Li, Qi Sun, Nathan Carr, Jonathan Ragan-Kelley, and Frédo Durand*
- [**(SIGGRAPH Asia 2019) High-Performance Computation on Sparse Data Structures**](http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[BibTex]](https://raw.githubusercontent.com/yuanming-hu/taichi/master/misc/taichi_bibtex.txt) [[Code]](https://github.com/taichi-dev/taichi)
  - by *Yuanming Hu, Tzu-Mao Li, Luke Anderson, Jonathan Ragan-Kelley, and Frédo Durand*

<div align="center">
  <img width="800px" src="https://github.com/taichi-dev/taichi/blob/master/docs/life_of_kernel_lowres.jpg">
</div>
