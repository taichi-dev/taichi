---
sidebar_position: 3
---

# Data-oriented Class

To define a Taichi kernel as a Python class member function:

1. Decorate the class with a `@ti.data_oriented` decorator.
2. Define `ti.kernel`s and `ti.func`s in your data-oriented Python class.

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
# a.call_inc() cannot be called, because a.temp has not been allocated at this point
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
        self.x = ti.field(dtype=ti.f32, shape=8)

    @ti.kernel
    def func(self, temp: ti.template()):
        for i in range(8):
            temp[i] = self.x[i * 2]

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


## Inheritance of data-oriented classes

The data-oriented property is automatically carried along with the Python class inheriting. This means that you can call a Taichi Kernel if any of its ancestor classes is decorated with `@ti.data_oriented`.

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
# The two lines above trigger a kernel define error, because class c is not decorated with @ti.data_oriented
```

## Python built-in decorators

Common decorators that are pre-built in Python, `@staticmethod`[^1] and `@classmethod`[^2], can decorate a Taichi kernel in data-oriented classes.

[^1]: [Python built-in functions - staticmethod](https://docs.python.org/3/library/functions.html#staticmethod)
[^2]: [Python built-in functions - classmethod](https://docs.python.org/3/library/functions.html#classmethod)


`staticmethod` example:

```python {16}
import taichi as ti

ti.init()

@ti.data_oriented
class Array2D:
    def __init__(self, n):
        self.arr = ti.Vector([0.] * n)

    @staticmethod
    @ti.func
    def clamp(x):  # Clamp to [0, 1)
        return max(0, min(1, x))
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
