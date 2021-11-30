# How to use CHI IR Builder

## Build Taichi

Follow the steps in `https://docs.taichi.graphics/lang/articles/contribution/dev_install`

Add option `-DTI_EXPORT_CORE=ON` to your `cmake` command (i.e. use `cmake .. DTI_EXPORT_CORE=ON`).

Set environment variable `$TAICHI_REPO_DIR` to your Taichi repository path.

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

## Reference

[IRBuilder test for external ptr](https://github.com/taichi-dev/taichi/blob/master/tests/cpp/ir/ir_builder_test.cpp#L87-L119) might be useful for those who want to run Taichi kernels in cpp within Taichi repository.
