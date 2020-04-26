# Changelog
- After v0.5.12: [[Releases]](https://github.com/taichi-dev/taichi/releases)
- (April 11, 2020) v0.5.11 released
   - **Automatic differentiation**
      - Fix floating-point type-cast gradients (#687) (by **Yuanming Hu**)
   - **CUDA backend**
      - PyPI package `taichi-nightly` now covers CUDA 10.X on Windows and Linux (#756) (by **Yuanming Hu**)
   - **Examples**
      - Add `game_of_life.py` (#741) (by **彭于斌**)
      - Fix `examples/regression.py` (#757) (by **Quan Wang**)
   - **GUI**
      - Support SPACE key (#749) (by **Ye Kuang**)
      - Fix blinking particles and random segmentation faults in `ti.GUI.circles` (#755) (by **Yuanming Hu**)
   - **Language and syntax**
      - Support `continue` on all backends (#716) (by **Ye Kuang**)
   - **LLVM backend (CPU and CUDA)**
      - Fix LLVM struct-for codegen crashing due to extra return #704 (#707) (by **Yuanming Hu**)
   - **Metal backend**
      - Support `ti.random()` on Metal (#710) (by **Ye Kuang**)
   - **OpenGL backend**
      - Support NVIDIA GLSL compiler (#666) (by **彭于斌**)
      - 64-bit data type support (#717) (by **彭于斌**)
      - Support more than one external array arguments (#694) (by **彭于斌**)
   - **IR and Optimization**
      - More Taichi IR standardization and optimization (#656) (by **xumingkuan**)
   - [Full log](https://github.com/taichi-dev/taichi/releases/tag/0.5.11)

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
- (Mar   3, 2020) v0.5.6 released
   - Fixed runtime LLVM bitcode loading failure on Linux
   - Fixed a GUI bug in `ti.GUI.line` (by **Mingkuan Xu [xumingkuan]**)
   - Fixed frontend syntax error false positive (static range-fors) (by **Mingkuan Xu [xumingkuan]**)
   - `arch=ti.arm64` is now supported. (Please build from source)
   - CUDA supported on NVIDIA Jetson. (Please build from source)
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
   - `@ti.classkernel` is deprecated. Always use `ti.kernel`, no matter you are decorating a class member function or not (by **Ye Kuang [k-ye]**)
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
- (Feb  12, 2020) v0.4.6 released.
   - (For compiler developers) An error will be raised when `TAICHI_REPO_DIR` is not a valid path (by **Yubin Peng [archibate]**)
   - Fixed a CUDA backend deadlock bug
   - Added test selectors `ti.require()` and `ti.archs_excluding()` (by **Ye Kuang [k-ye]**)
   - `ti.init(**kwargs)` now takes a parameter `debug=True/False`, which turns on debug mode if true
   - ... or use `TI_DEBUG=1` to turn on debug mode non-intrusively
   - Fixed `ti.profiler_clear`
   - Added `GUI.line(begin, end, color, radius)` and `ti.rgb_to_hex`
   -
   - Renamed `ti.trace` (Matrix trace) to `ti.tr`. `ti.trace` is now for logging with `ti.TRACE` level
   - Fixed return value of `ti test_cpp` (thanks to **Ye Kuang [k-ye]**)
   - Raise default loggineg level to `ti.INFO` instead of trace to make the world quiter
   - General performance/compatibility improvements
   - Doc updated
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
- (Jan  20, 2020) v0.3.25 released.
   - Experimental [CPU-only support for NVIDIA Jetson Nano](https://user-images.githubusercontent.com/34827518/72769070-62b1b200-3c34-11ea-8f6e-0f339b5b09ca.jpg) (with ARM CPUs. Building from source required.) (thanks to **Walter liu
 [hgnan]**)
- (Jan  19, 2020) v0.3.24 released.
   - `%` and `//` now follow Python semantics. Use `ti.raw_mod` for C-style `%` semantics (by **Chujie Zeng [Psycho7]**)
   - Parallel range-fors now supports non-compile-time constant bounds. For example, `for i in range(bound[0])` is supported
- (Jan  18, 2020) v0.3.23 released.
   - Taichi kernel calls now releases Python GIL
- (Jan  17, 2020) v0.3.22 released.
   - `ti.atomic_add()` now returns the old value (by **Ye Kuang [k-ye]**)
   - Experimental patch to Windows systems with malformed BIOS info (by **Chujie Zeng [Psycho7]**)
   - `ti.__version__` now returns the version triple, e.g. `(0, 3, 22)`
   - Fixed a CPU multithreading bug
   - Avoid accessor IR printing when setting `ti.cfg.print_ir = True`
   - Added `ti.cfg.print_accessor_ir`
   - Removed dependency on x86_64 SIMD intrinsics
   - Improved doc
- (Jan  11, 2020) v0.3.21 released.
   - GUI fix for OS X 10.14 and 10.15 (by **Ye Kuang [k-ye]**).
   - Minor improvements on documentation and profiler
- (Jan   2, 2020) v0.3.20 released.
   - Support `ti.static(ti.grouped(ti.ndrange(...)))`
- (Jan   2, 2020) v0.3.19 released.
   - Added `ti.atan2(y, x)`
   - Improved error msg when using float point numbers as tensor indices
- (Jan   1, 2020) v0.3.18 released.
   - Added `ti.GUI` class
   - Improved the performance of performance `ti.Matrix.fill`
- (Dec  31, 2019) v0.3.17 released.
   - Fixed cuda context conflict with PyTorch  (thanks to @Xingzhe He for reporting)
   - Support `ti.Matrix.T()` for transposing a matrix
   - Iteratable `ti.static(ti.ndrange)`
   - Fixed `ti.Matrix.identity()`
   - Added `ti.Matrix.one()` (create a matrix with 1 as all the entries)
   - Improved `ir_printer` on SNodes
   - Better support for `dynamic` SNodes.
     - `Struct-for's` on `dynamic` nodes supported
     - `ti.length` and `ti.append` to query and manipulate dynamic nodes
- (Dec  29, 2019) v0.3.16 released.
   - Fixed ndrange-fors with local variables (thanks to Xingzhe He for reporting this issue)
- (Dec  28, 2019) v0.3.15 released.
   - Multi-dimensional parallel range-for using `ti.ndrange`:
```python
  @ti.kernel
  def fill_3d():
    # Parallelized for all 3 <= i < 8, 1 <= j < 6, 0 <= k < 9
    for i, j, k in ti.ndrange((3, 8), (1, 6), 9):
      x[i, j, k] = i + j + k
```
- (Dec  28, 2019) v0.3.14 released.
   - GPU random number generator support for more than 1024x1024 threads
   - Parallelized element list generation on GPUs. Struct-fors significantly sped up.
   - `ti` and `tid` (debug mode) CLI commands
- (Dec  26, 2019) v0.3.13 released.
   - `ti.append` now returns the list length before appending
   - Fixed for loops with 0 iterations
   - Set `ti.get_runtime().set_verbose_kernel_launch(True)` to log kernel launches
   - Distinguish `/` and `//` following the Python convention
   - Allow using local variables as kernel argument type annotations
- (Dec  25, 2019) v0.3.11 released.
   - Support multiple kernels with the same name, especially in the OOP cases where multiple member kernels share the same name
   - Basic `dynamic` node support (`ti.append`, `ti.length`) in the new LLVM backend
   - Fixed struct-for loops on 0-D tensors
- (Dec  24, 2019) v0.3.10 released.
   - `assert <condition>` statement supported in Taichi kernels.
   - Comparison operator chaining (e.g. `1 < x <3`) supported in Taichi kernels.
- (Dec  24, 2019) v0.3.9 released.
   - `ti.classfunc` decorator for functions within a `data_oriented` class
   - `[Expr/Vector/Matrix].to_torch` now has a extra argument `device`, which specifies the device placement for returned torch tensor, and should have type `torch.device`. Default=`None`.
   - Cross-device (CPU/GPU) taichi/PyTorch interaction support, when using `to_torch/from_torch`.
   - #kernels compiled during external array IO significantly reduced (from `matrix size` to `1`)
- (Dec  23, 2019) v0.3.8 released.
   - **Breaking change**: `ti.data_oriented` decorator introduced. Please decorate all your Taichi data-oriented objects using this decorator. To invoke the gradient versions of `classmethod`, for example, `A.forward`, simply use `A.forward.grad()` instead of `A.forward(__gradient=True)` (obsolete).
- (Dec  22, 2019) v0.3.5 released.
   - Maximum tensor dimensionality is 8 now (used to be 4). I.e., you can now allocate up to 8-D tensors.
- (Dec  22, 2019) v0.3.4 released.
   - 2D and 3D polar decomposition (`R, S = ti.polar_decompose(A, ti.f32)`) and svd (`U, sigma, V = ti.svd(A, ti.f32)`) support. Note that `sigma` is a `3x3` diagonal matrix.
   - Fixed documentation versioning
   - Allow `expr_init` with `ti.core.DataType` as inputs, so that `ti.core.DataType` can be used as `ti.func` parameter
- (Dec  20, 2019) v0.3.3 released.
   - Loud failure message when calling nested kernels. Closed #310
   - `DiffTaichi` examples moved to [a standalone repo](https://github.com/yuanming-hu/difftaichi)
   - Fixed documentation versioning
   - Correctly differentiating kernels with multiple offloaded statements
- (Dec  18, 2019) v0.3.2 released
   - `Vector.norm` now comes with a parameter `eps` (`=0` by default), and returns `sqrt(\sum_i(x_i ^ 2) + eps)`. A non-zero `eps` safe guards the operator's gradient on zero vectors during differentiable programming.
- (Dec  17, 2019) v0.3.1 released.
   - Removed dependency on `glibc 2.27`
- (Dec  17, 2019) v0.3.0 released.
   - Documentation significantly improved
   - `break` statements supported in while loops
   - CPU multithreading enabled by default
- (Dec  16, 2019) v0.2.6 released.
   - `ti.GUI.set_image(np.ndarray/Taichi tensor)`
   - Inplace adds are *atomic* by default. E.g., `x[i] += j` is equivalent to `ti.atomic_add(x[i], j)`
   - `ti.func` arguments are forced to pass by value
   - `min/max` can now take more than two arguments, e.g. `max(a, b, c, d)`
   - Matrix operators `transposed`, `trace`, `polar_decompose`, `determinant` promoted to `ti` scope. I.e., users can now use `ti.transposed(M)` instead of `ti.Matrix.transposed(M)`
   - `ti.get_runtime().set_verbose(False)` to eliminate verbose outputs
   - LLVM backend now supports multithreading on CPUs
   - LLVM backend now supports random number generators (`ti.random(ti.i32/i64/f32/f64`)
- (Dec  5, 2019) v0.2.3 released.
   - Simplified interaction between `Taichi`, `numpy` and `PyTorch`
     - `taichi_scalar_tensor.to_numpy()/from_numpy(numpy_array)`
     - `taichi_scalar_tensor.to_torch()/from_torch(torch_array)`
- (Dec  4, 2019) v0.2.2 released.
   - Argument type `ti.ext_arr()` now takes PyTorch tensors
- (Dec  3, 2019) v0.2.1 released.
   - Improved type mismatch error message
   - native `min`/`max` supprt
   - Tensor access index dimensionality checking
   - `Matrix.to_numpy`, `Matrix.zero`, `Matrix.identity`, `Matrix.fill`
   - Warning instead of error on lossy stores
   - Added some initial support for cross-referencing local variables in different offloaded blocks.
- (Nov 28, 2019) v0.2.0 released.
   - More friendly syntax error when passing non-compile-time-constant values to `ti.static`
   - Systematically resolved the variable name resolution [issue](https://github.com/yuanming-hu/taichi/issues/282)
   - Better interaction with numpy:
     - `numpy` arrays passed as a `ti.ext_arr()` [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_numpy.py)
       - `i32/f32/i64/f64` data type support for numpy
       - Multidimensional numpy arrays now supported in Taichi kernels
     - `Tensor.to_numpy()` and `Tensor.from_numpy(numpy.ndarray)` supported [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_cvt_numpy.py)
     - Corresponding PyTorch tensor interaction will be supported very soon. Now only 1D f32 PyTorch tensors supproted when using `ti.ext_arr()`. Please use numpy arrays as intermediate buffers for now
   - Indexing arrays with an incorrect number of indices now results in a syntax error
   - Tensor shape reflection: [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_tensor_reflection.py)
     - `Tensor.dim()` to retrieve the dimensionality of a global tensor
     - `Tensor.shape()` to retrieve the shape of a global tensor
     - Note the above queries will cause data structures to be materialized
   - `struct-for` (e.g. `for i, j in x`) now supports iterating over tensors with non power-of-two dimensions
   - Handy tensor filling: [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_fill.py)
     - `Tensor.fill(x)` to set all entries to `x`
     - `Matrix.fill(x)` to set all entries to `x`, where `x` can be a scalar or `ti.Matrix` of the same size
   - Reduced python package size
   - `struct-for` with grouped indices for better metaprogramming, especially in writing dimensionality-independent code, in e.g. physical simulation: [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_grouped.py)
```python
for I in ti.grouped(x): # I is a vector of size x.dim() and data type i32
  x[I] = 0

# If tensor x is 2D
for I in ti.grouped(x): # I is a vector of size x.dim() and data type i32
  y[I + ti.Vector([0, 1])] = I[0] + I[1]
# is equivalent to
for i, j in x:
  y[i, j + 1] = i + j
```

- (Nov 27, 2019) v0.1.5 released.
   - [Better modular programming support](https://github.com/yuanming-hu/taichi/issues/282)
   - Disalow the use of `ti.static` outside Taichi kernels
   - Documentation improvements (WIP)
   - Codegen bug fixes
   - Special thanks to Andrew Spielberg and KLozes for bug report and feedback.
- (Nov 22, 2019) v0.1.3 released.
   - Object-oriented programming. [[Example]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_oop.py)
   - native Python function translation in Taichi kernels:
     - Use `print` instead of `ti.print`
     - Use `int()` instead of `ti.cast(x, ti.i32)` (or `ti.cast(x, ti.i64)` if your default integer precision is 64 bit)
     - Use `float()` instead of `ti.cast(x, ti.f32)` (or `ti.cast(x, ti.f64)` if your default float-point precision is 64 bit)
     - Use `abs` instead of `ti.abs`
     - Use `ti.static_print` for compile-time printing

- (Nov 16, 2019) v0.1.0 released. Fixed PyTorch interface.
- (Nov 12, 2019) v0.0.87 released.
   - Added experimental Windows support with a [[known issue]](https://github.com/yuanming-hu/taichi/issues/251) regarding virtual memory allocation, which will potentially limit the scalability of Taichi programs (If you are a Windows expert, please let me know how to solve this. Thanks!). Most examples work on Windows now.
   - CUDA march autodetection;
   - [Complex kernel](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_complex_kernels.py) to override autodiff.
 - (Nov 4, 2019) v0.0.85 released.
   - `ti.stop_grad` for stopping gradients during backpropagation. [[Example]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_stop_grad.py#L75);
   - Compatibility improvements on Linux and OS X;
   - Minor bug fixes.

<!---
| **Linux, Mac OS X** | **Windows** | Doc (WIP) | **Chat** |
|---------------------|------------------|----------------|------------------|
|[![Build Status](https://travis-ci.org/yuanming-hu/taichi.svg?branch=master)](https://travis-ci.org/yuanming-hu/taichi)|[![Build Status](https://ci.appveyor.com/api/projects/status/github/yuanming-hu/taichi?branch=master&svg=true)](https://ci.appveyor.com/project/IteratorAdvance/taichi)|[![Documentation Status](https://readthedocs.org/projects/taichi/badge/?version=latest)](http://taichi.readthedocs.io/en/latest/?badge=latest)|[![Join the chat at https://gitter.im/taichi-dev/Lobby](https://badges.gitter.im/taichi-dev/Lobby.svg)](https://gitter.im/taichi-dev/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)|
--->


## The Legacy Taichi Library [[Legacy branch]](https://github.com/yuanming-hu/taichi/tree/legacy)
The legacy **Taichi** library is an open-source computer graphics library written in C++14 and wrapped friendly with Python. It is no longer maintained since we have switched to the Taichi programming language and compiler.

## News
 - May 17, 2019: [Giga-Voxel SPGrid Topology Optimization Solver](https://github.com/yuanming-hu/spgrid_topo_opt) is released!
 - March 4, 2019: [MLS-MPM/CPIC solver](https://github.com/yuanming-hu/taichi_mpm) is now MIT-licensed!
 - August 14, 2018: [MLS-MPM/CPIC solver](https://github.com/yuanming-hu/taichi_mpm) reloaded! It delivers 4-14x performance boost over the previous state of the art on CPUs.

<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/topopt/bird-beak.gif">

 <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/water_wheel.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/sand_paddles.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/armodillo.gif" style=""> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/debris_flow.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/sand-sweep.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/sand_stir.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/bunny.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/robot_forward.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/banana.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/cheese.gif">

### [Getting Started (Legacy)](https://taichi.readthedocs.io/en/latest/installation.html#)
