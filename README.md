<div align="center">
  <img width="500px" src="https://github.com/yuanming-hu/taichi/raw/master/misc/logo.png">
  <h3> <a href="https://taichi.readthedocs.io/en/latest/"> Docs </a> | <a href="https://taichi.readthedocs.io/en/latest/hello.html"> Tutorial </a> | <a href="https://github.com/yuanming-hu/difftaichi"> DiffTaichi </a> | <a href="https://github.com/yuanming-hu/taichi/tree/master/examples"> Examples </a> | <a href="https://taichi.readthedocs.io/en/latest/contributor_guide.html"> Contribute </a> | <a href="https://forum.taichi.graphics/"> Forum </a> </h3>
</div>        

| **Documentations** | **Chat** | taichi-nightly | taichi-nightly-cuda-10-0 | taichi-nightly-cuda-10-1 |
|:-----|:-----|:----|:----|:----|
| [![Documentation Status](https://readthedocs.org/projects/taichi/badge/?version=latest)](http://taichi.readthedocs.io/en/latest/?badge=latest) | [![Join the chat at https://gitter.im/taichi-dev/Lobby](https://badges.gitter.im/taichi-dev/Lobby.svg)](https://gitter.im/taichi-dev/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) | [![Downloads](https://pepy.tech/badge/taichi-nightly/month)](https://pepy.tech/project/taichi-nightly/month)| [![Downloads](https://pepy.tech/badge/taichi-nightly-cuda-10-0/month)](https://pepy.tech/project/taichi-nightly-cuda-10-0/month) | [![Downloads](https://pepy.tech/badge/taichi-nightly-cuda-10-0/month)](https://pepy.tech/project/taichi-nightly-cuda-10-1/month) |

```bash
# Python 3.6/3.7 needed

# CPU only. No GPU/CUDA needed. (Linux, OS X and Windows)
python3 -m pip install taichi-nightly

# With GPU (CUDA 10.0) support (Linux only)
python3 -m pip install taichi-nightly-cuda-10-0

# With GPU (CUDA 10.1) support (Linux only)
python3 -m pip install taichi-nightly-cuda-10-1
```

|| **Linux (CUDA)** | **OS X (10.14+)** | **Windows** |
|:------|:-----|:-----|:-----|
|**Build**|[![Build Status](http://f11.csail.mit.edu:8080/job/taichi/badge/icon)](http://f11.csail.mit.edu:8080/job/taichi/)| [![Build Status](https://travis-ci.com/taichi-dev/taichi.svg?branch=master)](https://travis-ci.com/taichi-dev/taichi) | [![Build status](https://ci.appveyor.com/api/projects/status/yxm0uniin8xty4j7/branch/master?svg=true)](https://ci.appveyor.com/project/yuanming-hu/taichi/branch/master)|
|**PyPI**|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build status](https://ci.appveyor.com/api/projects/status/39ar9wa8yd49je7o?svg=true)](https://ci.appveyor.com/project/IteratorAdvance/taichi-wheels-test)|

## [Contribution Guidelines](https://taichi.readthedocs.io/en/latest/contributor_guide.html)

## Related papers
- [**(SIGGRAPH Asia 2019) High-Performance Computation on Sparse Data Structures**](http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[BibTex]](https://raw.githubusercontent.com/yuanming-hu/taichi/master/misc/taichi_bibtex.txt)
  - by *Yuanming Hu, Tzu-Mao Li, Luke Anderson, Jonathan Ragan-Kelley, and Frédo Durand*
- [**(ICLR 2020) Differentiable Programming for Physical Simulation**](https://arxiv.org/abs/1910.00935) [[Video]](https://www.youtube.com/watch?v=Z1xvAZve9aE) [[BibTex]](https://raw.githubusercontent.com/yuanming-hu/taichi/master/misc/difftaichi_bibtex.txt) [[Code]](https://github.com/yuanming-hu/difftaichi)
  - by *Yuanming Hu, Luke Anderson, Tzu-Mao Li, Qi Sun, Nathan Carr, Jonathan Ragan-Kelley, and Frédo Durand*

<div align="center">
  <img width="800px" src="https://github.com/taichi-dev/taichi/blob/master/docs/life_of_kernel_lowres.jpg">
</div>        

## Short-term goals
- (Done) Fully implement the LLVM backend to replace the legacy source-to-source C++/CUDA backends (By Dec 2019)
  - The only missing features compared to the old source-to-source backends:
    - Vectorization on CPUs. Given most users who want performance are using GPUs (CUDA), this is given low priority.
    - Automatic shared memory utilization. Postponed until Feb/March 2020.
- (Done) Redesign & reimplement (GPU) memory allocator (by the end of Jan 2020)
- (WIP) Tune the performance of the LLVM backend to match that of the legacy source-to-source backends (Hopefully by mid Feb, 2020. Current progress: setting up/tuning for final benchmarks)

## Updates
- (Feb   6, 2020) v0.4.5 released.
   - **`ti.init(arch=..., print_ir=..., default_fp=..., default_ip=...)`** now supported. `ti.cfg.xxx` is deprecated
   - **Immediate data layout specification** supported after `ti.init`. No need to wrap data layout definition with `@ti.layout` anymore (unless you intend to do so)
   - `ti.is_active`, `ti.deactivate`, `SNode.deactivate_all` supported in the new LLVM x64/CUDA backend. [Example](https://github.com/taichi-dev/taichi/blob/8b575a8ec2d8c7112191eef2a8316b793ba2452d/examples/taichi_sparse.py) <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/taichi/sparse_grids.gif">
   - Experimental [Windows non-UTF-8 path](https://github.com/taichi-dev/taichi/issues/428) fix (by **Yubin Peng [archibate]**)
   - `ti.global_var` (which duplicates `ti.var`) is removed
   - `ti.Matrix.rotation2d(angle)` added
- (Feb   5, 2020) v0.4.4 released.
   - For developers: [ffi-navigator](https://github.com/tqchen/ffi-navigator) support [[doc](https://taichi.readthedocs.io/en/latest/contributor_guide.html#efficient-code-navigation-across-python-c)]. (by **masahi**)
   - Fixed `f64` precision support of `sin` and `cos` on CUDA backends (by **Kenneth Lozes [KLozes]**)
   - Make Profiler print the arch name in its title (by **Ye Kuang [k-ye]**)
   - Tons of invisible contributions by **Ye Kuang [k-ye]**, for the WIP Metal backend
   - `Profiler` working on CPU devices. To enable, `ti.cfg.enable_profiler = True`. Call `ti.profiler_print()` to print kernel running times
   - General performance improvements
- (Feb   3, 2020) v0.4.3 released.
   - `GUI.circles` 2.4x faster
   - General performance improvements
- (Feb   2, 2020) v0.4.2 released.
   - GUI framerates are now more stable
   - Optimized OffloadedRangeFor with const bounds. Light computation programs such as `mpm88.py` is 30% faster on CUDA due to reduced kernel launches
   - Optimized CPU parallel range for performance
- (Jan  31, 2020) v0.4.1 released.
   - **Fixed an autodiff bug introduced in v0.3.24. Please update if you are using Taichi differentiable programming.**
   - Updated `Dockerfile` (by **Shenghang Tsai [jackalcooper]**)
   - `pbf2d.py` visualization performance boosted (by **Ye Kuang [k-ye]**)
   - Fixed `GlobalTemporaryStmt` codegen
- (Jan  30, 2020) v0.4.0 released.
   - Memory allocator redesigned
   - Struct-fors with pure dense data structures will be demoted into a range-for, which is faster since no element list generation is needed
   - Python 3.5 support is dropped. Please use Python 3.6(pip)/3.7(pip)/3.8(Windows: pip; OS X & Linux: build from source) (by **Chujie Zeng [Psycho7]**)
   - `ti.deactivate` now supported on sparse data structures
   - `GUI.circles` (batched circle drawing) performance improved by 30x
   - Minor bug fixes (by **Yubin Peng [archibate], Ye Kuang [k-ye]**)
   - Doc updated

- [Full changelog](changelog.md)
