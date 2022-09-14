---
sidebar_position: 5
---

# Glossary

## Abstract syntax tree (AST)

<https://en.wikipedia.org/wiki/Abstract_syntax_tree>

## Compound type

A compound type is a user-defined array-like or struct-like data type which comprises multiple members of primitive types or other compound types. Supported compound types in Taichi include vectors, metrics, and structs.

## Coordinate offset

A coordinate offset refers to a value added to another base value, which is an element in a Taichi field.

You can use offsets when defining a [field](#field) to move the field boundaries.

## External array

External arrays refer to data containers available in the Python scope.

Taichi supports interaction with the following external arrays - Numpy arrays, PyTorch tensors, and Paddle tensors.

## Field

A field is a multi-dimensional array of elements. The elements it accepts can be a scalar, a vector, a matrix, or a struct. It is a global data container provided by Taichi and can be accessed from both the [*Python scope*](#python-scope) and the [*Taichi scope*](#taichi-scope).

## Field shape

The shape of a field is the number of elements in each dimension.

## Kernel

A kernel is a function decorated with `@ti.kernel`. A kernel serves as the entry point where Taichi begins to take over the tasks, and it must be called directly by Python code.

## Megakernel

A megakernel is a Taichi kernel that can deal with a large amount of computation to achieve high arithmetic intensity.

## Metadata

Metadata refers to the two fundamental attributes of a Taichi field, i.e., data type and shape.

Use `field.dtype` and `field.shape` to retrieve the metadata.

## Primitive type

Primitive data types are commonly-used numerical data types from which all other data types are constructed. Supported primitive data types in Taichi include `ti.i32` (`int32`), `ti.u8` (`uint8`), and `ti.f64` (`float64`)

## Python scope

Code outside of the Taichi scope is in the Python scope. The code in the Python scope is native Python and executed by Python's virtual machine, not by Taichi's runtime.

The Python scope corresponds to the host side in CUDA.

## Sparse matrix

A matrix is a two-dimensional data object made of m rows and n columns. If a matrix is comprised of mostly zero values, then it is a sparse matrix.

Taichi provides *APIs* for sparse matrices.

## Static scope

A static scope is a scope of the argument of `ti.static`, which is a hint for the compiler to evaluate the argument at compile time.

## Taichi function

A Taichi function is a function decorated with `@ti.func`.

A Taichi function must be called from inside a kernel or from inside another Taichi function.

## Taichi scope

The code inside a kernel or a Taichi function is in the Taichi scope. The code in the Taichi scope is compiled by Taichi's runtime and executed in parallel on CPU or GPU devices for high-performance computation.

The Taichi scope corresponds to the device side in CUDA.

## Template signature

Template signatures are what distinguish different instantiations of a kernel template.

For example, The signature of `add(x, 42)` is `(x, ti.i32)`, which is the same as that of `add(x, 1)`. Therefore, the latter can reuse the previously compiled binary. The signature of `add(y, 42)` is `(y, ti.i32)`, a different value from the previous signature, hence a new kernel will be instantiated and compiled.
