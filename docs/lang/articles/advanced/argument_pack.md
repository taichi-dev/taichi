---
sidebar_position: 6
---

# Taichi Argument Pack

Taichi provides custom [argpack types](../type_system/type.md#argument-pack-type) for developers to cache unchanged parameters between multiple kernel calls.

Argument packs, also known as argpacks, are user-defined data types that act as wrappers for parameters. They allow multiple parameters to be stored and used as a single parameter. One key advantage of using argpacks is their ability to buffer parameters. If you have certain parameters that remain unchanged when calling kernels, you can store them in argpacks. Taichi can then cache these argpacks on device, resulting in improved kernel performance.

## Creation and Initialization

You can use the function `ti.types.argpack()` to create an argpack type, which can be utilized to represent view params. The following is an example of defining a Taichi argument pack:

```python
view_params_tmpl = ti.types.argpack(view_mtx=ti.math.mat4, proj_mtx=ti.math.mat4, far=ti.f32)
```

You can then use this `view_params_tmpl` to initialize an argument pack with given values.

```python cont
view_params = view_params_tmpl(
    view_mtx=ti.math.mat4(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]),
    proj_mtx=ti.math.mat4(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]),
    far=1)
```

## Pass Argument Packs to Kernels

Once argument packs are created and initialized, they can be easily used as kernel parameters. Simply pass them to the kernel, and Taichi will intelligently cache them (if their values remain unchanged) across multiple kernel calls, optimizing performance.

```python cont
@ti.kernel
def p(view_params: view_params_tmpl) -> ti.f32:
    return view_params.far


print(p(view_params))  # 1.0
```

## How does it Work

### Caching parameter values

Argument packs facilitate the caching of parameters within the device buffer, leading to significant performance enhancements for kernels that are repeatedly invoked with identical parameters.

- In the absence of argument packs, parameter values are copied to device memory every time, leading to additional time overhead caused by the memory copying process.
  ![Copying Operations Performed Without ArgPacks](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/doc/without_argpack_memory_copying.svg)

- By employing argument packs, parameter values can be stored and cached directly in the device buffer, eliminating the need for redundant and resource-consuming memory copying operations.
  ![Copying Operations Performed With ArgPacks](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/doc/argument_pack_memory_copying.svg)


### Argpack Nesting

:::note
This subsection delves into implementation specifics related to argpack, which may not be relevant or necessary for Taichi users to understand or utilize in their own workflows. Therefore, Taichi users can safely skip reading this section as it does not directly impact their usage or requirements.
:::

To enhance usability and convenience, argpacks have the ability to be nested within other argpacks. This feature allows for the organization and encapsulation of multiple argument packs within a single parent argpack. By enabling nesting, users can efficiently manage and handle complex sets of arguments, improving code clarity and maintainability.

To enable nested argpacks, the parent argpack creates a pointer to the argpack buffer of the nested argpack. During code generation, a recursive process is implemented to load argpack buffers from parent argpacks to child argpacks.

For example, consider that we want to load a value in `Value #3.1`. We first load Arguments Buffer, then load the pointer to argpack #1. Then we load pointer to argpack #3 inside argpack buffer #1. Finally we load value #3.1 in argpack buffer #3.

![Load Value 3.1 in Nested ArgPacks](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/doc/argpack_nesting_structure.svg)

Value #3.1 is a struct value containing a child struct and an integer. The child struct contains an integer and a float. Consider that we'd like to load this float value. This diagram illustrates the steps to load this float. It also explains `arg_id` and `arg_depth` in detail.

![Load Float Value in Value 3.1](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/doc/argload_stmt_for_argpack_nesting.svg)


### Buffer Values in Argpack

:::note
This subsection delves into implementation specifics related to argpack, which may not be relevant or necessary for Taichi users to understand or utilize in their own workflows. Therefore, Taichi users can safely skip reading this section as it does not directly impact their usage or requirements.
:::

In its implementation, an argpack is specifically designed to store constant values like primitive types, matrices, and dataclasses directly within its buffer. However, for resource types that require more intricate handling, such as Ndarrays, external arrays, sparse matrices, and textures, argpack employs a different approach. Instead of storing these resources directly in the argpack buffer, it temporarily holds them within the Python `ArgPack` class. During kernel launching, argpack passes the pointers to these resources by storing them in the argument buffer.

## Limitations

Argpacks are primarily designed to serve as a parameter cache, which naturally introduces certain limitations on their usage:

- Argpacks can only be used as kernel parameters.
- Argpacks cannot be used as return types.
- While argpacks can be nested within other argpacks, they cannot be nested within Compound Types.
- Presently, certain types within argpacks are read-only when used in kernels, but their values can be modified outside of kernels.
  - Constant Values: These include primitive types, matrices, and dataclasses (structs). They are passed by value, making them read-only in kernels.
  - Resources: This category encompasses Ndarrays, external arrays, sparse matrices, and textures. They are passed by pointers to buffers, allowing both reading and writing operations in kernels.
