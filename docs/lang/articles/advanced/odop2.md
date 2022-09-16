---
sidebar_position: 3
---

# Objective Data-oriented Programming II


## Taichi dataclasses

Taichi provides custom [struct types](../type_system/type.md#compound-types) for developers to assemble pieces of data together. However, it would be more convenient to have:
  1. A Python representation of the struct type which is more object oriented.
  2. Functions associated with a struct type (C++-style structs).


To achieve these two points, developers can use the `@ti.dataclass` decorator on a Python class.  This is heavily inspired by the Python [dataclass](https://docs.python.org/3/library/dataclasses.html) feature, which uses class fields with annotations to create data types.

### Creating a struct from a Python class
Here is an example of how we could create a Taichi struct type from a Python class:

```python
@ti.dataclass
class Sphere:
    center: vec3
    radius: ti.f32
```
This will create the *exact* same type as using `ti.types.struct()`:

```python
Sphere = ti.types.struct(center=vec3, radius=ti.f32)
```
The `@ti.dataclass` decorator converts the annotated members in the Python class to members in the resulting struct type. In both of the above examples, you end up with the same struct field.

```python
sphere_field = Sphere.field(shape=(n,))
```

### Associating functions with the struct type
Python classes can have functions attached to them, and so can Taichi struct types. Building from the above example, one can embed functions in the struct as follows:

```python
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

Functions associated with structs follow the same [scope rules](../kernels/syntax.md#taichi-scope-vs-python-scope) as normal functions, in that they can be in Taichi or Python scope.  Each instance of the `Sphere` struct type now will have the above functions added to them.  The functions can be called such as:

```python
a_python_struct = Sphere(center=vec3(0.0), radius=1.0)
# calls a python scope function from python
a_python_struct.is_zero_sized() # False

@ti.kernel
def get_area() -> ti.f32:
    a_taichi_struct = Sphere(center=vec3(0.0), radius=4.0)
    # return the area of the sphere, a taichi scope function
    return a_taichi_struct.area()
get_area() # 201.062...
```

### Notes
- Inheritance of Taichi dataclasses is not supported.
- While it is convenient and recommended to associate functions with a struct defined via `@ti.dataclass`, `ti.types.struct` can serve the same purpose with the help of the `__struct_methods` argument. As mentioned above, the two methods of defining a struct type produce identical output.

```python
@ti.func
def area(self):
    # a function to run in taichi scope
    return 4 * math.pi * self.radius * self.radius

Sphere = ti.types.struct(center=vec3, radius=ti.f32,
                         __struct_methods={'area': area})
```
