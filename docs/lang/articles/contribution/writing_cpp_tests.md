---
sidebar_position: 11
---

# Workflow for writing a CPP test

We strongly recommend each developer to write CPP unit tests when sending a PR.

- We use [googletest](https://github.com/google/googletest) for as the CPP
  test infrastructure.
- CPP tests should be added to the `tests/cpp/` directory.

## Build and run Taichi CPP tests

```bash
# inside build/
cmake .. -DTI_BUILD_TESTS=ON # ... other regular Taichi cmake args
make

# run the CPP test
./taichi_cpp_tests
```

:::note
Consider polishing the CPP test infrastructure:

* Separate each translation unit into its own test executable
* Have a unified script to control the execution of which set of tests
:::

## Adding a new test case

Please follow [Googletest Primer](https://google.github.io/googletest/primer.html) and [Advanced googletest Topics](https://google.github.io/googletest/advanced.html).