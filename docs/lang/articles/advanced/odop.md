---
sidebar_position: 2
---

# Objective Data-oriented Programming

Taichi is a
[data-oriented](https://en.wikipedia.org/wiki/Data-oriented_design)
programming (DOP) language. However, simple DOP makes modularization
hard.

To allow modularized code, Taichi borrow some concepts from
object-oriented programming (OOP).

For convenience, let's call the hybrid scheme **objective data-oriented
programming** (ODOP).


## Data-oriented classes

### Introduction

If you need to define a **Taichi kernel** as a Python class member function, please decorate the class with a `@ti.data_oriented` decorator. You can then define `ti.kernel`s and `ti.func`s in your *data-oriented* Python class.

:::note
The first argument of the function should be the class instance ("`self`"), unless you are defining a `@staticmethod`.
:::

A brief example:

```python {1}
@ti.data_oriented
class TiArray:
    def __init__(self, n):
        self.x = ti.field(dtype=ti.i32, shape=n)

    @ti.kernel
    def inc(self):
        for i in self.x:
            self.x[i] += 1

a = TiArray(32)
a.inc()
```

Definitions of Taichi fields can be made not only in _init_ functions, but also at any place of a Python-scope function in a data-oriented class. For example,

```python {21,25}
import taichi as ti

ti.init()

@ti.data_oriented
class MyClass:
    @ti.kernel
    def inc(self, temp: ti.template()):
        for I in ti.grouped(temp):
            temp[I] += 1

    def call_inc(self):
        self.inc(self.temp)

    def allocate_temp(self, n):
        self.temp = ti.field(dtype = ti.i32, shape=n)


a = MyClass()
# a.call_inc() cannot be called, since a.temp has not been allocated at this point
a.allocate_temp(4)
a.call_inc()
a.call_inc()
print(a.temp)  # [2 2 2 2]
a.allocate_temp(8)
a.call_inc()
print(a.temp)  # [1 1 1 1 1 1 1 1]
```

Another memory recycling example:
```python
import taichi as ti

ti.init()

@ti.data_oriented
class Calc:
    def __init__(self):
        self.x = ti.field(dtype=ti.f32, shape=16)
        self.y = ti.field(dtype=ti.f32, shape=4)

    @ti.kernel
    def func(self, temp: ti.template()):
        for i in range(8):
            temp[i] = self.x[i * 2] + self.x[i * 2 + 1]

        for i in range(4):
            self.y[i] = max(temp[i * 2], temp[i * 2 + 1])

    def call_func(self):
        fb = ti.FieldsBuilder()
        temp = ti.field(dtype=ti.f32)
        fb.dense(ti.i, 8).place(temp)
        tree = fb.finalize()
        self.func(temp)
        tree.destroy()


a = Calc()
for i in range(16):
    a.x[i] = i
a.call_func()
print(a.y)  # [ 5. 13. 21. 29.]
```


### Inheritance of data-oriented classes

The *data-oriented* property will be automatically carried beyond the Python class inheriting. This means the **Taichi Kernel** could be called while any of the ancestor classes are decorated by the `@ti.data_oriented` decorator.

An example:
```python
import taichi as ti

ti.init(arch=ti.cuda)

class BaseClass:
    def __init__(self):
        self.n = 10
        self.num = ti.field(dtype=ti.i32, shape=(self.n, ))

    @ti.kernel
    def count(self) -> ti.i32:
        ret = 0
        for i in range(self.n):
            ret += self.num[i]
        return ret

    @ti.kernel
    def add(self, d: ti.i32):
        for i in range(self.n):
            self.num[i] += d


@ti.data_oriented
class DataOrientedClass(BaseClass):
    pass

class DeviatedClass(DataOrientedClass):
    @ti.kernel
    def sub(self, d: ti.i32):
        for i in range(self.n):
            self.num[i] -= d


a = DeviatedClass()
a.add(1)
a.sub(1)
print(a.count())  # 0


b = DataOrientedClass()
b.add(2)
print(b.count())  # 1

c = BaseClass()
# c.add(3)
# print(c.count())
# The two lines above will trigger a kernel define error, since class c is not decorated by @ti.data_oriented
```

### Python built-in decorators

Common decorators that are pre-built in Python, `@staticmethod`[^1] and `@classmethod`[^2], could decorate to a **Taichi kernel** in *data-oriented* classes.

[^1]: [Python built-in functions - staticmethod](https://docs.python.org/3/library/functions.html#staticmethod)
[^2]: [Python built-in functions - classmethod](https://docs.python.org/3/library/functions.html#classmethod)


`staticmethod` example :

```python {16}
import taichi as ti

ti.init()

@ti.data_oriented
class Array2D:
    def __init__(self, n, m, increment):
        self.n = n
        self.m = m
        self.val = ti.field(ti.f32)
        self.total = ti.field(ti.f32)
        self.increment = float(increment)
        ti.root.dense(ti.ij, (self.n, self.m)).place(self.val)
        ti.root.place(self.total)

    @staticmethod
    @ti.func
    def clamp(x):  # Clamp to [0, 1)
        return max(0., min(1 - 1e-6, x))

    @ti.kernel
    def inc(self):
        for i, j in self.val:
            ti.atomic_add(self.val[i, j], self.increment)

    @ti.kernel
    def inc2(self, increment: ti.i32):
        for i, j in self.val:
            ti.atomic_add(self.val[i, j], increment)

    @ti.kernel
    def reduce(self):
        for i, j in self.val:
            ti.atomic_add(self.total[None], self.val[i, j] * 4)

arr = Array2D(2, 2, 3)

double_total = ti.field(ti.f32, shape=())

ti.root.lazy_grad()

arr.inc()
arr.inc.grad()
print(arr.val[0, 0])  # 3
arr.inc2(4)
print(arr.val[0, 0])  # 7

with ti.ad.Tape(loss=arr.total):
    arr.reduce()

for i in range(arr.n):
    for j in range(arr.m):
        print(arr.val.grad[i, j])  # 4

@ti.kernel
def double():
    double_total[None] = 2 * arr.total[None]

with ti.ad.Tape(loss=double_total):
    arr.reduce()
    double()

for i in range(arr.n):
    for j in range(arr.m):
        print(arr.val.grad[i, j])  # 8
```

`classmethod` example:
```python {12}
import taichi as ti

ti.init(arch=ti.cuda)

@ti.data_oriented
class Counter:
    num_ = ti.field(dtype=ti.i32, shape=(32, ))
    def __init__(self, data_range):
        self.range = data_range
        self.add(data_range[0], data_range[1], 1)

    @classmethod
    @ti.kernel
    def add(cls, l: ti.i32, r: ti.i32, d: ti.i32):
        for i in range(l, r):
            cls.num_[i] += d

    @ti.kernel
    def num(self) -> ti.i32:
        ret = 0
        for i in range(self.range[0], self.range[1]):
            ret += self.num_[i]
        return ret

a = Counter((0, 5))
print(a.num())  # 5
b = Counter((4, 10))
print(a.num())  # 6
print(b.num())  # 7
```

## Taichi dataclasses

Taichi provides custom [struct types](../type_system/type.md#compound-types) for developers to associate pieces of data together. However, it is often convenient to have:
  1. A Python representation of the struct type which is more object oriented.
  2. Functions associated with a struct type. (C++ style structs)


To achieve these two points, developers can use the `@ti.dataclass` decorator on a Python class.  This is heavily inspired by the Python [dataclass](https://docs.python.org/3/library/dataclasses.html) feature, which uses class fields with annotations to create data types.

### Creating a struct from a Python class
Here is an example of how we could create a Taichi struct type from a Python class:

```python
@ti.dataclass
class Sphere:
    center: vec3
    radius: ti.f32
```
This will create the *exact* same type as doing:

```python
Sphere = ti.types.struct(center=vec3, radius=ti.f32)
```
Using the `@ti.dataclass` decorator will convert the annotated fields in the Python class to members in the resulting struct type.  In both of the above examples you would create a field of the struct the same way.

```python
sphere_field = Sphere.field(shape=(n,))
```

### Associating functions with the struct type
Python classes can have functions attached to them, as can Taichi struct types.  Building from the above example, here is how one would add functions to the struct.

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
- Inheritance of Taichi dataclasses is not implemented.
- While functions attached to a struct with the `@ti.dataclass` decorator is convenient and encouraged, it is actually possible to associate a function to structs with the older method of defining structs.  As mentioned above, the two methods for defining a struct type are identical in their output.  To do this, use the `__struct_methods` argument with the `ti.types.struct` call:

```python
@ti.func
def area(self):
    # a function to run in taichi scope
    return 4 * math.pi * self.radius * self.radius

Sphere = ti.types.struct(center=vec3, radius=ti.f32,
                         __struct_methods={'area': area})
```
