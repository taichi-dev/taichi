# How to use CHI IR Builder

## Build Taichi

Follow the steps in `https://docs.taichi.graphics/lang/articles/contribution/dev_install`

Add option `-DTI_EXPORT_CORE=ON` to your `cmake` command (i.e. use `cmake .. DTI_EXPORT_CORE=ON`).

Make sure taichi is built under `$TAICHI_REPO_DIR/build` directory.

After building, `$TAICHI_REPO_DIR/build/libtaichi_export_core.so` should exist.

## Link with the Taichi Shared Library

`main.cpp` shows how to construct and run Taichi kernels using CHI IR Builder.

```bash
mkdir build
cd build
cmake ..
make
./chi_examples
```
