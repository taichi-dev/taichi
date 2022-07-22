---
sidebar_position: 4
---

# Write a C++ test

We strongly recommend each developer to write C++ unit tests when sending a PR.

- We use [googletest](https://github.com/google/googletest) as the C++
  test infrastructure.
- C++ tests should be added to the `tests/cpp/` directory.
- Make sure your C++ test source file is covered by [this CMake glob](https://github.com/taichi-dev/taichi/blob/fb4741421ca79e971852464ffdf0ff066e667c92/cmake/TaichiTests.cmake#L13-L23).

## Build and run Taichi C++ tests

```bash
# build taichi with tests enabled
TAICHI_CMAKE_ARGS="-DTI_BUILD_TESTS:BOOL=ON" python setup.py develop

# run the C++ test
TI_LIB_DIR=$TAICHI_INSTALL_DIR/_lib/runtime ./taichi_cpp_tests
```

:::note
`$TAICHI_INSTALL_DIR` can be retrieved from `taichi.__path__[0]` in python.
:::

:::note
On Windows, `taichi_cpp_tests.exe` will be placed inside the `%TAICHI_REPO_DIR%\bin` directory.
:::

:::note
Consider polishing the C++ test infrastructure:

* Separate each translation unit into its own test executable
* Have a unified script to control the execution of which set of tests
:::

## Add a new test case

Please follow [Googletest Primer](https://google.github.io/googletest/primer.html) and [Advanced googletest Topics](https://google.github.io/googletest/advanced.html).
