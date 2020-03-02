<div align="center">
  <img width="500px" src="https://github.com/yuanming-hu/taichi/raw/master/misc/logo.png">
  <h3> <a href="https://taichi.readthedocs.io/en/latest/"> Docs </a> | <a href="https://taichi.readthedocs.io/en/latest/hello.html"> Tutorial </a> | <a href="https://github.com/yuanming-hu/difftaichi"> DiffTaichi </a> | <a href="https://github.com/yuanming-hu/taichi/tree/master/examples"> Examples </a> | <a href="https://taichi.readthedocs.io/en/latest/contributor_guide.html"> Contribute </a> | <a href="https://forum.taichi.graphics/"> Forum </a> </h3>
</div>        

| **Documentations** | **Chat** | taichi-nightly | taichi-nightly-cuda-10-0 | taichi-nightly-cuda-10-1 |
|:-----|:-----|:----|:----|:----|
| [![Documentation Status](https://readthedocs.org/projects/taichi/badge/?version=latest)](http://taichi.readthedocs.io/en/latest/?badge=latest) | [![Join the chat at https://gitter.im/taichi-dev/Lobby](https://badges.gitter.im/taichi-dev/Lobby.svg)](https://gitter.im/taichi-dev/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) | [![Downloads](https://pepy.tech/badge/taichi-nightly)](https://pepy.tech/project/taichi-nightly) | [![Downloads](https://pepy.tech/badge/taichi-nightly-cuda-10-0)](https://pepy.tech/project/taichi-nightly-cuda-10-0) | [![Downloads](https://pepy.tech/badge/taichi-nightly-cuda-10-1)](https://pepy.tech/project/taichi-nightly-cuda-10-1) |

```bash
# Python 3.6/3.7 needed for all platforms. Python 3.8 supported only on OS X and Windows

# CPU only. No GPU/CUDA needed. (Linux, OS X and Windows)
python3 -m pip install taichi-nightly

# With GPU (CUDA 10.0) support (Linux only)
python3 -m pip install taichi-nightly-cuda-10-0

# With GPU (CUDA 10.1) support (Linux only)
python3 -m pip install taichi-nightly-cuda-10-1

# Build from source if you work in other environments
```

## [Contribution Guidelines](https://taichi.readthedocs.io/en/latest/contributor_guide.html)

|| **Linux (CUDA)** | **OS X (10.14+)** | **Windows** |
|:------|:-----|:-----|:-----|
|**Build**|[![Build Status](http://f11.csail.mit.edu:8080/job/taichi/badge/icon)](http://f11.csail.mit.edu:8080/job/taichi/)| [![Build Status](https://travis-ci.com/taichi-dev/taichi.svg?branch=master)](https://travis-ci.com/taichi-dev/taichi) | [![Build status](https://ci.appveyor.com/api/projects/status/yxm0uniin8xty4j7/branch/master?svg=true)](https://ci.appveyor.com/project/yuanming-hu/taichi/branch/master)|
|**PyPI**|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build Status](https://travis-ci.com/yuanming-hu/taichi-wheels-test.svg?branch=master)](https://travis-ci.com/yuanming-hu/taichi-wheels-test)|[![Build status](https://ci.appveyor.com/api/projects/status/39ar9wa8yd49je7o?svg=true)](https://ci.appveyor.com/project/IteratorAdvance/taichi-wheels-test)|

## Updates
- (Mar   2, 2020) v0.5.5 released: **Experimental CUDA 10.0/10.1 support on Windows. Feedbacks are welcome!**
- (Mar   1, 2020) v0.5.4 released
   - Metal backend now supports < 32bit args (#530) (by **Ye Kuang [k-ye]**)
   - Added `ti.imread/imwrite/imshow` for convenient image IO (by **Yubin Peng [archibate]**)
   - `ti.GUI.set_image` now takes all numpy unsigned integer types (by **Yubin Peng [archibate]**)
   - Bug fix: [Make sure KernelTemplateMapper extractors's size is the same as the number of args](https://github.com/taichi-dev/taichi/issues/534) (by **Ye Kuang [k-ye]**)
   - [Avoid duplicate evaluations in chaining comparison (such as `1 < ti.append(...) < 3 < 4`)](https://github.com/taichi-dev/taichi/issues/540) (by **Mingkuan Xu [xumingkuan]**)
   - Frontend kernel/function structure checking (#544) (by **Mingkuan Xu [xumingkuan]**)
   - Throw exception instead of SIGABRT to obtain RuntimeError in Python-scope (by **Yubin Peng [archibate]**)
   - Mark sync bit only after running a kernel on GPU (by **Ye Kuang [k-ye]**)
   - Fix ti.func AST transform (due to locals() not saving compile result) #538, #539 (by **Yubin Peng [archibate]**)
   - Add a KernelSimplicityASTChecker to ensure grad kernel is compliant (#553) (by **Ye Kuang [k-ye]**)
   - Fixed MSVC C++ mangling which leads to unsupported characters in LLVM NVPTX ASM printer
   - CUDA unified memory dependency is now removed. Set `TI_USE_UNIFIED_MEMORY=0` to disable unified memory usage
   - Improved `ti.GUI.line` performance
   - (For developers) compiler significantly refactored and folder structure reorganized
- (Feb  25, 2020) v0.5.3 released
   - Better error message when try to declare tensors after kernel invocation (by **Yubin Peng [archibate]**)
   - Logging: `ti.warning` renamed to `ti.warn`
   - Arch: `ti.x86_64` renamed to `ti.x64`. `ti.x86_64` is deprecated and will be removed in a future release
   - (For developers) Improved runtime bit code compilation thread safety (by **Yubin Peng [archibate]**)
   - Improved OS X GUI performance (by **Ye Kuang [k-ye]**)
   - Experimental support for new integer types `u8, i8, u16, i16, u32` (by **Yubin Peng [archibate]**)
   - Update doc (by **Ye Kuang [k-ye]**)
- (Feb  20, 2020) v0.5.2 released
   - Gradients for `ti.pow` now supported (by **Yubin Peng [archibate]**)
   - Multi-threaded unit testing (by **Yubin Peng [archibate]**)
   - Fixed Taichi crashing when starting multiple instances simultaneously (by **Yubin Peng [archibate]**)
   - Metal backend now supports `ti.pow` (by **Ye Kuang [k-ye]**)
   - Better algebraic simplification (by **Mingkuan Xu [xumingkuan]**)
   - `ti.normalized` now optionally takes a argument `eps` to prevent division by zero in differentiable programming
   - Improved random number generation by decorrelating PRNG streams on CUDA
   - Set environment variable `TI_LOG_LEVEL` to `trace`, `debug`, `info`, `warn`, `error` to filter out/increase verbosity. Default=`info`
   - [bug fix] fixed a loud failure on differentiable programming code generation due to a new optimization pass
   - Added `ti.GUI.triangle` [example](https://github.com/taichi-dev/taichi/blob/master/misc/test_gui.py#L11)
   - Doc update: added `ti.cross` for 3D cross products
   - Use environment variable `TI_TEST_THREADS` to override testing threads
   - [For Taichi developers, bug fix] `ti.init(print_processed=True)` renamed to `ti.init(print_preprocessed=True)`
   - Various development infrastructure improvements by **Yubin Peng [archibate]**
   - Official Python3.6 - Python3.8 packages on OS X (by **wYw [Detavern]**)
- (Feb  16, 2020) v0.5.1 released
   - Keyboard and mouse events supported in the GUI system. Check out [mpm128.py](https://github.com/taichi-dev/taichi/blob/4f5cc09ae0e35a47ad71fdc582c1ecd5202114d8/examples/mpm128.py) for a interactive demo! (by **Yubin Peng [archibate] and Ye Kuang [k-ye]**)
   - Basic algebraic simplification passes (by **Mingkuan Xu [xumingkuan]**)
   - (For developers) `ti` (`ti.exe`) command supported on Windows after setting `%PATH%` correctly (by **Mingkuan Xu [xumingkuan]**)
   - General power operator `x ** y` now supported in Taichi kernels (by **Yubin Peng [archibate]**)
   - `.dense(...).pointer()` now abbreviated as `.pointer(...)`. `pointer` now stands for a dense pointer array. This leads to cleaner code and better performance. (by **Kenneth Lozes [KLozes]**)
   - (Advanced struct-fors only) `for i in X` now iterates all child instances of `X` instead of `X` itself. Skip this if you only use `X=leaf node` such as `ti.f32/i32/Vector/Matrix`.
   - Fixed cuda random number generator racing conditions
- (Feb  14, 2020) **v0.5.0 released with a new Apple Metal GPU backend for Mac OS X users!** (by **Ye Kuang [k-ye]**)
   - Just initialize your program with `ti.init(..., arch=ti.metal)` and run Taichi on your Mac GPUs!
   - A few takeaways if you do want to use the Metal backend:
     - For now, the Metal backend only supports `dense` SNodes and 32-bit data types. It doesn't support `ti.random()` or `print()`.
     - Pre-2015 models may encounter some undefined behaviors under certain conditions (e.g. read-after-write). According to our tests, it seems like the memory order on a single GPU thread could go inconsistent on these models.
     - The `[]` operator in Python is slow in the current implementation. If you need to do a large number of reads, consider dumping all the data to a `numpy` array via `to_numpy()` as a workaround. For writes, consider first generating the data into a `numpy` array, then copying that to the Taichi variables as a whole.
     - Do NOT expect a performance boost yet, and we are still profiling and tuning the new backend. (So far we only saw a big performance improvement on a 2015 MBP 13-inch model.)
- [Full changelog](changelog.md)

## Short-term goals
- (Done) Fully implement the LLVM backend to replace the legacy source-to-source C++/CUDA backends (By Dec 2019)
  - The only missing features compared to the old source-to-source backends:
    - Vectorization on CPUs. Given most users who want performance are using GPUs (CUDA), this is given low priority.
    - Automatic shared memory utilization. Postponed until Feb/March 2020.
- (Done) Redesign & reimplement (GPU) memory allocator (by the end of Jan 2020)
- (WIP) Tune the performance of the LLVM backend to match that of the legacy source-to-source backends (Hopefully by Feb, 2020. Current progress: setting up/tuning for final benchmarks)

## Related papers
- [**(SIGGRAPH Asia 2019) High-Performance Computation on Sparse Data Structures**](http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf) [[Video]](https://youtu.be/wKw8LMF3Djo) [[BibTex]](https://raw.githubusercontent.com/yuanming-hu/taichi/master/misc/taichi_bibtex.txt)
  - by *Yuanming Hu, Tzu-Mao Li, Luke Anderson, Jonathan Ragan-Kelley, and Frédo Durand*
- [**(ICLR 2020) Differentiable Programming for Physical Simulation**](https://arxiv.org/abs/1910.00935) [[Video]](https://www.youtube.com/watch?v=Z1xvAZve9aE) [[BibTex]](https://raw.githubusercontent.com/yuanming-hu/taichi/master/misc/difftaichi_bibtex.txt) [[Code]](https://github.com/yuanming-hu/difftaichi)
  - by *Yuanming Hu, Luke Anderson, Tzu-Mao Li, Qi Sun, Nathan Carr, Jonathan Ragan-Kelley, and Frédo Durand*

<div align="center">
  <img width="800px" src="https://github.com/taichi-dev/taichi/blob/master/docs/life_of_kernel_lowres.jpg">
</div>
