---
sidebar_position: 6
---

# Objective data-oriented programming

Taichi is a
[data-oriented](https://en.wikipedia.org/wiki/Data-oriented_design)
programming (DOP) language. However, simple DOP makes modularization
hard.

To allow modularized code, Taichi borrow some concepts from
object-oriented programming (OOP).

For convenience, let's call the hybrid scheme **objective data-oriented
programming** (ODOP).


## Data-oriented class

If you need to define a **Taichi kernel** as a Python class member function, you need to decorate the class with a `@ti.data_oriented` decorator. Then, you can define `ti.kernel`s and `ti.func`s in your *data-oriented* Python class.

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
            self.x[i] += 1.0

a = TiArray(32)
a.inc()
```

## Some walkaround about Python

### Inherit of data-oriented class

The *data-oriented* property will be automatically carried beyond the Python class inheriting. Which means that, the kernels could be called while any of the ancestor class is decorated by `@ti.data_oriented` decorator.

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
print(a.count())


b = DataOrientedClass()
b.add(2)
print(b.count())

c = BaseClass()
# c.add(3)
# print(c.count())
# These two lines would cause error...
```


### Python-built-in-decorators

Common decorators that are pre-built in Python, `@staticmethod` and `@classmethod`, could decorate to a **Taichi kernel** in *data-oriented* class.

:::note 
`@property` decorator is not supported now. Would be fixed soon.
:::

`staticmethod` example :

```python
import taichi as ti

ti.init()

@ti.data_oriented
class Array2D:
    def __init__(self, n, m, increment):
        self.n = n
        self.m = m
        self.val = ti.field(ti.f32)
        self.total = ti.field(ti.f32)
        self.increment = increment
        ti.root.dense(ti.ij, (self.n, self.m)).place(self.val)
        ti.root.place(self.total)

    @staticmethod
    @ti.func
    def clamp(x):  # Clamp to [0, 1)
        return max(0, min(1 - 1e-6, x))

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
            ti.atomic_add(self.total, self.val[i, j] * 4)

arr = Array2D(128, 128, 3)

double_total = ti.field(ti.f32, shape=())

ti.root.lazy_grad()

arr.inc()
arr.inc.grad()
assert arr.val[3, 4] == 3
arr.inc2(4)
assert arr.val[3, 4] == 7

with ti.Tape(loss=arr.total):
    arr.reduce()

for i in range(arr.n):
    for j in range(arr.m):
        assert arr.val.grad[i, j] == 4

@ti.kernel
def double():
    double_total[None] = 2 * arr.total

with ti.Tape(loss=double_total):
    arr.reduce()
    double()

for i in range(arr.n):
    for j in range(arr.m):
        assert arr.val.grad[i, j] == 8
```

`classmethod` example:
```python
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
b = Counter((4, 10))
print(b.num())
```
