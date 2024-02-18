---
sidebar_position: 1
---

# Math Module


Taichi provides a built-in `math` module that supports frequently used mathematical functions and utility functions, including:

- Commonly-used mathematical functions that are analogous to those in Python's built-in `math` module.
- Small vector and matrix types that are analogous to those in the [OpenGL shading language](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language) (GLSL).
- Some GLSL-standard functions.
- Complex number operations in the form of 2D vectors.

## Mathematical functions

You must call the mathematical functions provided by Taichi's `math` module from within the Taichi scope. For example:

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


These functions also take vectors and matrices as arguments and operate on them element-wise:

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

:::note


Taichi's math module overlaps to a large extent with Python's built-in math module. Ensure that you follow a few extra rules when using Taichi's math module:

- You must call the functions provided by Taichi's math module from within the Taichi scope.
- Functions in Taichi's math module also take vectors or matrices as arguments.
- The precision of a function in Taichi's math module depends on the settings of `default_fp` and `arch` (backend) in `ti.init()`.


:::

## Small vector and matrix types


Taichi's math module provides a few small vector and matrix types:


- `vec2/vec3/vec4`: 2D/3D/4D floating-point vector types.
- `ivec2/ivec3/ivec4`: 2D/3D/4D integer vector types.
- `uvec2/uvec3/uvec4`: 2D/3D/4D unsigned integer vector types.
- `mat2/mat3/mat4`: 2D/3D/4D floating-point square matrix types.

To create one of the vector/matrix types above, use template function `ti.types.vector()` or `ti.types.matrix()`. For example, `vec2` is defined as follows:

```python
vec2 = ti.types.vector(2, float)
```


The number of precision bits of such a type is determined by `default_fp` or `default_ip` in the `ti.init()` method call. For example, if `ti.init(default_fp=ti.f64)` is called, then `vec2/vec3/vec4` and `mat2/mat3/mat4` defined in the Taichi scope all have a 64-bit floating-point precision.

You can use these types to instantiate vectors/matrices or annotate data types for function arguments and struct members. See the [Type System](../type_system/type.md) for more information. Here we emphasize that they have very flexible initialization routines:


```python
mat2 = ti.math.mat2
vec3 = ti.math.mat3
vec4 = ti.math.vec4

m = mat2(1)  # [[1., 1.], [1., 1.]]
m = mat2(1, 2, 3, 4)  # [[1., 2.], [3, 4.]]
m = mat2([1, 2], [3, 4])  # [[1., 2.], [3, 4.]]
m = mat2([1, 2, 3, 4])  # [[1., 2.], [3, 4.]]
v = vec3(1, 2, 3)
m = mat2(v, 4)  # [[1., 2.], [3, 4.]]
u = vec4([1, 2], [3, 4])
u = vec4(v, 4.0)
```

Another important feature of vector types created by `ti.types.vector()` is that they support **vector swizzling** just as GLSL vectors do. This means you can use `xyzw`, `rgba`, `stpq` to access their elements with indices &le; four:


```python
v = ti.math.vec4(1, 2, 3, 4)
u = v.xyz  # vec3(1, 2, 3)
u = v.xxx  # vec3(1, 1, 1)
u = v.wzyx  # vec4(4, 3, 2, 1)
u = v.rraa  # vec4(1, 1, 2, 2)
```


### Relations between `ti.Vector`, `ti.types.vector` and `ti.math.vec3`

- `ti.Vector` is a function that accepts a 1D array and returns a matrix instance that has only one column. For example, `ti.Vector([1, 2, 3, 4, 5])`.
- `ti.types.vector` is a function that accepts an integer and a primitive type and returns a vector type. For example: `vec5f = ti.types.vector(5, float)`. `vec5f` can then be used to instantiate 5D vectors or annotate data types of function arguments and struct members:
    ```python
    @ti.kernel
    def test(v: vec5f):
        print(v.xyz)
    ```

    Unlike `ti.Vector`, whose input data must be a 1D array, vector types created by `ti.types.vector()` have more flexible ways to initialize, as explained above.

- `ti.math.vec3` is created by `vec3 = ti.types.vector(3, float)`.



## GLSL-standard functions


Taichi's math module also supports a few [GLSL standard functions](https://registry.khronos.org/OpenGL-Refpages/gl4/index.php). These functions follow the GLSL standard, except that they accept arbitrary vectors and matrices as arguments and operate on them element-wise. For example:

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


## Complex number operations


Taichi's math module also supports basic complex arithmetic operations on 2D vectors.

You can use a 2D vector of type `ti.math.vec2` to represent a complex number. In this way, additions and subtractions of complex numbers come in the form of 2D vector additions and subtractions. You can call `ti.math.cmul()` and `ti.math.cdiv()` to conduct multiplication and division of complex numbers:


```python
import taichi as ti
import taichi.math as tm
ti.init()

@ti.kernel
def test():
    x = tm.vec2(1, 1)  # complex number 1+1j
    y = tm.vec2(0, 1)  # complex number 1j
    z = tm.cmul(x, y)  # vec2(-1, 1) = -1+1j
    w = tm.cdiv(x, y)  #  vec2(1, -1) = 1-1j
```

You can also compute the power, logarithm, and exponential of a complex number:


```python
@ti.kernel
def test():
    x = tm.vec2(1, 1)  # complex number 1 + 1j
    y = tm.cpow(x, 2)  # complex number (1 + 1j)**2 = 2j
    z = tm.clog(x)     # complex number (0.346574 + 0.785398j)
    w = tm.cexp(x)     # complex number (1.468694 + 2.287355j)
```
