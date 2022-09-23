# The math module


Taichi implements its own `math` module to support usual mathematical functions plus a few utility functions that are frequently used in computer graphics and simulations. They include:

1. Commonly-used mathematical functions that are analogous to those in Python's built-in `math` module.
2. Small vector and matrix types that are analogous to those in the [OpenGL shading language](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language)(GLSL).
3. Some GLSL-standard functions.
4. Complex number arithmetic of 2D vectors.

Details will follow below.


## Mathematical functions

Currently, Taichi's `math` module covers a large portion of funtions in Python's built-in `math` module. You need to call them in the Taichi scope. For example:

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

These functions also accept vectors and matrices as arguments, and they apply to them element-wise:

```python
@ti.kernel
def test():
    a = ti.Vector([1.0, 2.0, 3.0])
    x = tm.sin(a)  # [0.841471, 0.909297, 0.141120]
    y = tm.floor(a)  #  [1.000000, 2.000000, 3.000000]
    z = tm.degrees(a)  #  [57.295780, 114.591560, 171.887344]
    b = ti.Vector([2.0, 3.0, 4.0])
    w = tm.atan2(b, a)  # [1.107149, 0.982794, 0.927295]
    ...
```

## Small vector and matrix types

Taichi's math module supplies a few small vector and matrix types:


1. `vec2/vec3/vec4` for 2D/3D/4D floating-point vector types.
2. `ivec2/ivec3/ivec4` for 2D/3D/4D integer vector types.
3. `uvec2/uvec3/uvec4` for 2D/3D/4D unsigned integer vector types.
4. `mat2/mat3/mat4` for 2D/3D/4D floating-point square matrix types.


Indeed, these vector/matrix types are created using the two template functions `ti.types.vector()` and `ti.types.matrix()`. For example the `vec2` type is defined in the following way:

```python
vec2 = ti.types.vector(2, float)
```

The number of precision bits of these matrix types will be determined by `default_fp` and `default_ip` in your `ti.init()` call. For example, if `ti.init(default_fp=ti.f64)` is called, then the types `vec2/vec3/vec4` and `mat2/mat3/mat4` will all have 64-bit floating-point precision.

These types can be used to instantiate vectors and matrices or annotate the data types of function arguments and struct members. See [type system](../type_system/type.md) for more detail. Here we emphasize that they have very flexible initialization routines:

```python
mat2 = ti.math.mat2
vec3 = ti.math.mat3

m = mat2(1)  # [[1., 1.], [1., 1.]]
m = mat2(1, 2, 3, 4)  # [[1., 2.], [3, 4.]]
m = mat2([1, 2], [3, 4])  # [[1., 2.], [3, 4.]]
m = mat2([1, 2, 3, 4])  # [[1., 2.], [3, 4.]]
v = vec3(1, 2, 3)
m = mat2(v, 4)  # [[1., 2.], [3, 4.]]
```


## GLSL-standard functions


Taichi's math module also supports a few [GLSL standard functions](https://registry.khronos.org/OpenGL-Refpages/gl4/index.php), they are implemented in the way that follows the GLSL standard, except that they accept arbitrary vectors and matrices as arguments, and apply to them element-wise. For example:

```python
import taichi as ti
import taichi.math as tm

@ti.kernel
def example():
    v = tm.vec3(0., 1., 2.)
    w = tm.smoothstep(0.0, 1.0, v)
    w = tm.clamp(w, 0.2, 0.8)
    w = tm.reflect(v, tm.normalize(tm.vec3(1)))
```

:::note

Texture support in Taichi is implemented in the `ti.types.texture_types` module.

:::


## Complex number arithmetic

Taichi's math module also supports basic complex arithmetic in the form of 2D vectors.

You can use a 2D vector of type `ti.math.vec2` to represent a complex number, the addtion/subtraction of complex numbers are just addtion/subtraction of 2D vectors, and multiplication and division can be performed by calling the two functions `ti.math.cmul` and `ti.math.cdiv`:

```python
import taichi as ti
import taichi.math as tm
ti.init()

@ti.kernel
def test():
    x = tm.vec2(1, 1)  # complex number 1+1j
    y = tm.vec2(0, 1)  # complex number 1j
    z = tm.cmul(x, y)  # vec2(-1, 1) = -1+1j
    w = tm.cdiv(x, y)  #  vec2(2, 0) = 2+0j
```

You can also compute the power, logarithm and exponetial of a complex number:

```python

@ti.kernel
def test():
    x = tm.vec2(1, 1)
    y = tm.cpow(x, 2)
    z = tm.clog(x)
    w = tm.cexp(x)
```
