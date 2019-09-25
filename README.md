# DiffSim Programming Language
### Differentiable Physical Simulation

# Installation
Supports Ubuntu 14.04/16.04/18.04, ArchLinux, Mac OS X. For GPU support, CUDA 9.0+ is needed.

 - Execute `python3 -m pip install astpretty astor pytest opencv-python pybind11==2.2.4`
 - Install `taichi` with the [installation script](https://taichi.readthedocs.io/en/latest/installation.html#ubuntu-arch-linux-and-mac-os-x). **Checkout branch `llvm-debug`**.
 - Put this repo under `taichi/projects/`
 - Add the following line to your `~/.bashrc` or `~/.zshrc` for the python frontend.
 ```bash
 export PYTHONPATH=$TAICHI_REPO_DIR/projects/taichi_lang/python:$PYTHONPATH
 ```
 - Execute `source ~/.bashrc` (or `source ~/.zshrc`) to reload shell config.
 - Execute `ti build` to build.
 - Execute `ti test` to run all the tests. It may take a around 20 minutes to run all tests.
 - Check out `examples` for runnable DiffSim examples. Run them with `python3`.

# Folder Structure
Key folders are
 - *examples* : example programs written in Taichi
   - *cpp*: benchmarking examples in the SIGGRAPH Asia paper (mpm_benchmark.cpp, smoke_renderer.cpp, cnn.cpp)
   - *fem*: the FEM benchmark
 - *include*: language runtime
 - *src*: the compiler implementation (The functionality is briefly documented in each file)
   - *analysis*: static analysis passes
   - *backends*: codegen to x86 and CUDA
   - *transforms*: IR transform passes
   - *ir*: the intermediate representation system
   - *program*: the context for taichi programs
   - ...
 - *test*: unit tests

# Troubleshooting
 - Run with debug mode to see if there's any illegal memory access;
 - Disable compiler optimizations to quickly confirm that the issue is not cause by optimization;
