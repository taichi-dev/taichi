---
sidebar_position: 6
---

# Taichi Argument Pack

Taichi provides custom [argpack types](../type_system/type.md#argument-pack-type) for developers to cache unchanged parameters between multiple kernel calls.

Argument packs, also known as argpacks, are user-defined data types that act as wrappers for parameters. They allow multiple parameters to be stored and used as a single parameter. One key advantage of using argpacks is their ability to buffer parameters. If you have certain parameters that remain unchanged when calling kernels, you can store them in argpacks. Taichi can then cache these argpacks, resulting in improved program performance by making it faster.

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

## Limitations

Argpacks are not commonly used types. They are primarily designed as parameter containers, which naturally impose certain limitations on their usage:

- Argpacks can only be used as kernel parameters
- Argpacks cannot be used as return types
- Argpacks cannot be nested in Compound Types, but can be nested in other argpacks.

:::note

While argument pack interfaces are currently supported in Taichi, the internal caching mechanism is still being developed. This feature is planned to be implemented in future versions of Taichi.

:::
