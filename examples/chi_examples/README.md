# How to use CHI IR Builder

## Build Taichi

Follow the steps in `https://taichi.readthedocs.io/en/stable/dev_install.html`.

Add option `-DTI_EXPORT_CORE=ON` to your `cmake` command (i.e. use `cmake .. DTI_EXPORT_CORE=ON`).

Make sure taichi is built under `$TAICHI_REPO_DIR/build` directory.

After building, `$TAICHI_REPO_DIR/build/libtaichi_export_core.so` should exist.

## link taichi core to use CHI IR Builder

`main.cpp` shows how to construct and run Taichi kernels using CHI IR Builder.

`CMakeLists.txt` gives necessary compiling flags and include directories when linking CHI IR Builder.

### build and run

```bash
mkdir build
cd build
cmake ..
make
./chi_examples
```
