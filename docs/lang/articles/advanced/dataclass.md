---
sidebar_position: 4
---

# Taichi Dataclass

Taichi provides custom [struct types](../type_system/type.md#compound-types) for developers to assemble pieces of data together. However, it would be more convenient to have:

- A Python representation of the struct type which is more Object-Oriented.
- Functions associated with a struct type.


To achieve the ends, Taichi enabled the `@ti.dataclass` decorator on a Python class. This is inspired by Python's [dataclass](https://docs.python.org/3/library/dataclasses.html) feature, which uses class fields with annotations to create data types.

:::note

The `dataclass` in Taichi is simply a wrapper for `ti.types.struct`. Therefore, the member types that a `dataclass` object can contain are the same as those allowed in a struct. They must be one of the following types: scalars, matrix/vector types, and other dataclass/struct types. Objects like `field`, `Vector field`, and `Ndarray` cannot be used as members of a `dataclass` object.

:::

## Create a struct from a Python class

The following is an example of defining a Taichi struct type under a Python class:

```python
vec3 = ti.math.vec3

@ti.dataclass
class Sphere:
    center: vec3
    radius: ti.f32
```
This is the same equivalent as using `ti.types.struct()`:

```python cont
Sphere = ti.types.struct(center=vec3, radius=ti.f32)
```
The `@ti.dataclass` decorator converts the annotated members in the *Python class* to members in the resulting *struct type*. In both of the above examples, you end up with the same struct field.


## Associate Functions with the struct type

Both Python classes and Taichi struct types can have functions attached to them. Building from the above example, one can embed functions in the struct as follows:

```python
vec3 = ti.math.vec3

@ti.dataclass
class Sphere:
    center: vec3
    radius: ti.f32

    @ti.func
    def area(self):
        # a function to run in taichi scope
        return 4 * math.pi * self.radius * self.radius

    def is_zero_sized(self):
        # a python scope function
        return self.radius == 0.0
```

Functions associated with structs follow the same scope rules as other functions. In other words, they can be placed in either the Taichi scope or the Python scope. Each instance of the `Sphere` struct type now have the above functions attached to them. The functions can be called in the following way:

```python {3,10} cont
a_python_struct = Sphere(center=ti.math.vec3(0.0), radius=1.0)
# calls a python scope function from python
a_python_struct.is_zero_sized() # False

@ti.kernel
def get_area() -> ti.f32:
    a_taichi_struct = Sphere(center=ti.math.vec3(0.0), radius=4.0)
    # return the area of the sphere, a taichi scope function
    return a_taichi_struct.area()
get_area() # 201.062...
```

## Notes

- Inheritance of Taichi dataclasses is not supported.
- Default values in Taichi dataclasses are not supported.
- While it is convenient and recommended to associate functions with a struct defined via `@ti.dataclass`, `ti.types.struct` can serve the same purpose with the help of the `__struct_methods` argument. As mentioned above, the two methods of defining a struct type produce identical output.

```python
@ti.func
def area(self):
    # a function to run in taichi scope
    return 4 * math.pi * self.radius * self.radius

Sphere = ti.types.struct(center=ti.math.vec3, radius=ti.f32,
                         __struct_methods={'area': area})
```
