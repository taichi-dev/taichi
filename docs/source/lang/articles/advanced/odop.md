---
sidebar_position: 2
---

# Objective Data-Oriented Programming

Taichi is a [Data-Oriented](https://en.wikipedia.org/wiki/Data-Oriented_design) Programming (DOP) language. However, one-size-fits-all DOP makes modularization hard. To allow modularized code, Taichi borrows some concepts from Object-Oriented Programming (OOP). For convenience, let's call the hybrid scheme Objective Data-Oriented Programming (ODOP).

:::note
DOP approaches coding in a unique way. While you may be familiar with OOP, the Data-Oriented design indicates that everything is data that can be acted on. This differentiates functionality from data. They are no longer linked by a set of rules. Your DOP routines are general-purpose and deal with enormous volumes of data. To guarantee that the function takes as little effort as possible, you should organize the data as close to the output data as possible.
:::

The ODOP scheme allows you to organize data and methods in a class and call the methods to manipulate the data in the Taichi scope. Taichi offers two different types of classes that serve this purpose, and they are distinguished by the two decorators `@ti.data_oriented` and `@ti.dataclass` respectively:

- Decorated with `@ti.data_oriented`, a Data-Oriented class is used when your data is actively updated in the Python scope (such as current time and user input events) and tracked in Taichi kernels. This type of class can have native Python objects as members and must be instantiated in the Python scope. [Data-Oriented Class](./data_oriented_class.md) describes this type of class.
- Decorated with `@ti.dataclass`, a dataclass is a wrapper of `ti.types.struct`. A dataclass provides more flexibilities. You can define Taichi functions as its methods and call these methods in the Taichi scope. [Taichi Dataclass](./dataclass.md) describes this type of class.
