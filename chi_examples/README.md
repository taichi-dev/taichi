# How to use CHI IR Builder

## Build Taichi

Follow the steps in `https://docs.taichi.graphics/lang/articles/contribution/dev_install`

Add option `-DTI_EXPORT_CORE=ON` when building Taichi (As an example, run `TAICHI_CMAKE_ARGS="-DTI_EXPORT_CORE=ON" python3 setup.py install --user`).

## Link with the Taichi Shared Library

`main.cpp` shows how to construct and run Taichi kernels using CHI IR Builder.

```bash
mkdir build
cd build
cmake ..
make
./chi_examples
```

If you want to build CHI examples in a non-default folder, set environment variable `$TAICHI_REPO_DIR` to your Taichi repository path.

## Reference

[IRBuilder test for external ptr](https://github.com/taichi-dev/taichi/blob/master/tests/cpp/ir/ir_builder_test.cpp#L87-L119) might be useful for those who want to run Taichi kernels in cpp within Taichi repository.
