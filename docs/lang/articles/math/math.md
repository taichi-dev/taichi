# The math module


Taichi implements its own `math` module to support usual mathematical functions plus a few utility functions that are frequently used in computer graphics and simulations. They include:

1. Mathematical functions analogous to those in Python's built-in `math` module.
2. Small vector and matrix types analogous to those in the [OpenGL shading language](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language)(GLSL), for example `vec2/vec3/vec4` and `mat2/mat3/mat4`.
3. Functions analogous to those built-in functions in GLSL.
4. Complex number arithmetic of 2D vectors.

Details will follow below.


## Mathematical functions

Currently, Taichi's `math` module covers a large portion of funtions in Python's built-in `math` module, except that these functions are implemented following the C-style, and you need to call them in the Taichi scope. For example:

```python
import taichi as ti
import taichi.math as tm
ti.init()

@ti.kernel
def test():
    a = 1.0
    x = tm.sin(a)
    y = tm.floor(a)
    z = tm.degrees(a)
    w = tm.log2(a)
    ...
```

These function also accept vectors and matrices as arguments, and they apply to them element-wise:

```python
@ti.kernel
def test():
    a = ti.Vector([1.0, 2.0, 3.0])
    x = tm.sin(a)
    y = tm.floor(a)
    z = tm.degrees(a)
    b = ti.Vector([2.0, 3.0, 4.0])
    w = tm.atan2(b, a)
    ...
```

## Small vector and matrix types

Taichi's math module supplies a few small vector and matrix types:


1. `vec2/vec3/vec4` for 2D/3D/4D floating-point vector types.
2. `ivec2/ivec3/ivec4` for 2D/3D/4D integer vector types.
3. `uvec2/uvec3/uvec4` for 2D/3D/4D unsigned integer vector types.
4. `mat2/mat3/mat4` for 2D/3D/4D floating-point square matrix types.

The precision bits of these matrix types will be determined by the settings for `default_fp` and `default_ip` in your `ti.init()` call.


These types can be used to instantiate vectors and matrices or annotate the data types of function arguments and struct members. See [type system](../type_system/type.md) for more detail.

Indeed, these vector/matrix types are created using the two template functions `ti.types.vector()` and `ti.types.matrix()`, for example

```python

```