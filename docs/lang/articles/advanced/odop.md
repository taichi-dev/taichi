---
sidebar_position: 2
---

# Objective Data-oriented Programming

Taichi is a [data-oriented](https://en.wikipedia.org/wiki/Data-oriented_design) programming (DOP) language. However, one-size-fits-all DOP makes modularization hard. To allow modularized code, Taichi borrows some concepts from object-oriented programming (OOP). For convenience, let's call the hybrid scheme objective data-oriented programming (ODOP).

The ODOP scheme allows you to organize data and methods in a class and call the methods to manipulate the data in the Taichi scope. Taichi offers two different types of classes that serve this purpose, and they are distinguished by the two decorators `@ti.data_oriented` and `@ti.dataclass` respectively:

- Decorated with `@ti.data_oriented`, a data-oriented class is used when your data is actively updated in the Python scope (such as current time and user input events) and tracked in Taichi kernels. This type of class can have native Python objects as members and must be instantiated in the Python scope. [Data-oriented Class](./data_oriented_class.md) describes this type of class.
- Decorated with `@ti.dataclass`, a dataclass is a wrapper of `ti.types.struct`. A dataclass provides more flexibilities. You can define Taichi functions as its methods and call these methods in the Taichi scope. [Data Class](./dataclass.md) describes this type of class.
