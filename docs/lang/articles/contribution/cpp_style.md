---
sidebar_position: 4
---

# C++ style

We generally follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). One major exception is the naming convention of functions: Taichi adopts the snake case for function naming, as opposed to the camel case [suggested in Google's style](https://google.github.io/styleguide/cppguide.html#Function_Names), e.g. `this_is_a_taichi_function()`.

Below we highlight some of the most widely used styles.

## Naming conventions

- Class and struct names should use the camel case, e.g. `CodegenLlvm`.
  - Prefer capitalizing only the first letter of an acronym/abbreviation ([examples](https://google.github.io/styleguide/jsguide.html#naming-camel-case-defined)).
- Variable names should use the snake case, e.g. `llvm_context`.
- Private class member variable names should end with an `_`, e.g. `id_to_snodes_`.
- Constant names should use the camel case, with a prefix `k`, e.g. `constexpr int kTaichiMaxNumArgs = 64;`.
- Macros should start with `TI_`, e.g. `TI_NOT_IMPLEMENTED`.
  - In general, avoid using macros as much as possible.
  - Avoid using `TI_NAMESPACE_BEGIN/END` in the new code.

## Rule of thumbs

- Use `const` as much as possible, e.g. function parameter types, class member functions, etc.
- Provide default initializers to the class member variables, at least for the POD types.
  ```cpp
  class Foo {
   private:
    int x_{0};
    char* buf_{nullptr};
  };
  ```
- Embrace the smart pointers and avoid `new` and `delete`.
- Mark the constructor `explicit` to prevent the compiler from doing any implicit conversion.
- Avoid virtual function calls in the constructors or destructors ([explanation](https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP50-CPP.+Do+not+invoke+virtual+functions+from+constructors+or+destructors)).
