---
sidebar_position: 5
---

# Glossary

## Abstract syntax tree (AST)

<https://en.wikipedia.org/wiki/Abstract_syntax_tree>

## Ahead-of-time (AOT)

<https://en.wikipedia.org/wiki/Ahead-of-time_compilation>

## Aliasing

<https://en.wikipedia.org/wiki/Aliasing_(computing)>

## Annotation

<https://docs.python.org/3/glossary.html>

## Array of structures (AOS)

<https://en.wikipedia.org/wiki/AoS_and_SoA>

See also [structure of arrays](./glossary.md/#structure-of-arrays-soa).

## Assert statement

<https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement>

## Atomic operation

<https://wiki.osdev.org/Atomic_operation>

## Augmented assignment

<https://en.wikipedia.org/wiki/Augmented_assignment>

## Automatic differentiation

<https://en.wikipedia.org/wiki/Automatic_differentiation>

## Bitmask

<https://en.wikipedia.org/wiki/Mask_(computing)>

## Column-major order

<https://en.wikipedia.org/wiki/Row-_and_column-major_order>

See also [row-major order](./glossary.md/#row-major-order).

## Compound type

A compound type is a user-defined array-like or struct-like data type which comprises multiple members of primitive types or other compound types. Supported compound types in Taichi include vectors, metrics, and structs.

## Compute shader

<https://www.khronos.org/opengl/wiki/Compute_Shader>

## Coordinate offset

A coordinate offset refers to a value added to another base value, which is an element in a Taichi field.

You can use offsets when defining a [field](#field) to move the field boundaries.

## Data-oriented programming (DOP)

<https://en.wikipedia.org/wiki/Data-oriented_design>

## Data race

<https://en.wikipedia.org/wiki/Race_condition#Data_race>

## Differentiable programming

<https://en.wikipedia.org/wiki/Differentiable_programming>

## Domain-specific language (DSL)

<https://en.wikipedia.org/wiki/Domain-specific_language>

## External array

External arrays refer to data containers available in the Python scope.

Taichi supports interaction with the following external arrays - Numpy arrays, PyTorch tensors, and Paddle tensors.

## Field

A field is a multi-dimensional array of elements. The elements it accepts can be a scalar, a vector, a matrix, or a struct. It is a global data container provided by Taichi and can be accessed from both the [*Python scope*](#python-scope) and the [*Taichi scope*](#taichi-scope).

## Field shape

The shape of a field is the number of elements in each dimension.

## Fragment shader

<https://en.wikipedia.org/wiki/Shader#Pixel_shaders>

## Global variable

<https://en.wikipedia.org/wiki/Global_variable>

## Grid-Stride Loop

<https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/>

## Imperative programming

<https://en.wikipedia.org/wiki/Imperative_programming>

## Instantiation

<https://en.wikipedia.org/wiki/Instance_(computer_science)>

## Intermediate representation (IR)

<https://en.wikipedia.org/wiki/Intermediate_representation>

## Just-in-time (JIT) compilation

<https://en.wikipedia.org/wiki/Just-in-time_compilation>

## Kernel

A kernel is a function decorated with `@ti.kernel`. A kernel serves as the entry point where Taichi begins to take over the tasks, and it must be called directly by Python code.

## Lexical-scoped

<https://en.wikipedia.org/wiki/Scope_(computer_science)#Lexical_scope>

## Local variable

<https://en.wikipedia.org/wiki/Local_variable>

## Loop unrolling

<https://en.wikipedia.org/wiki/Loop_unrolling>

## Megakernel

A megakernel is a Taichi kernel that can deal with a large amount of computation to achieve high arithmetic intensity.

## Metadata

Metadata refers to the two fundamental attributes of a Taichi field, i.e., data type and shape.

Use `field.dtype` and `field.shape` to retrieve the metadata.

## Metaprogramming

<https://en.wikipedia.org/wiki/Metaprogramming>

## Object-oriented programming (OOP)

<https://en.wikipedia.org/wiki/Object-oriented_programming>

## Plain old data (POD)

<https://en.wikipedia.org/wiki/Passive_data_structure>

## Primitive type

Primitive data types are commonly-used numerical data types from which all other data types are constructed. Supported primitive data types in Taichi include `ti.i32` (`int32`), `ti.u8` (`uint8`), and `ti.f64` (`float64`)

## Python scope

Code outside of the Taichi scope is in the Python scope. The code in the Python scope is native Python and executed by Python's virtual machine, not by Taichi's runtime.

The Python scope corresponds to the host side in CUDA.

## Row-major order

<https://en.wikipedia.org/wiki/Row-_and_column-major_order>

See also [coloum-major order](./glossary.md/#column-major-order).

## Shader storage buffer object (SSBO)

<https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object>

## Sparse matrix

A matrix is a two-dimensional data object made of m rows and n columns. If a matrix is comprised of mostly zero values, then it is a sparse matrix.

Taichi provides *APIs* for sparse matrices.

## Static scope

A static scope is a scope of the argument of `ti.static`, which is a hint for the compiler to evaluate the argument at compile time.

## Static single assignment (SSA)

<https://en.wikipedia.org/wiki/Static_single-assignment_form>

## Structure of arrays (SOA)

<https://en.wikipedia.org/wiki/AoS_and_SoA>

See also [array of structures](./glossary.md/#array-of-structures-aos).

## Taichi function

A Taichi function is a function decorated with `@ti.func`.

A Taichi function must be called from inside a kernel or from inside another Taichi function.

## Taichi scope

The code inside a kernel or a Taichi function is in the Taichi scope. The code in the Taichi scope is compiled by Taichi's runtime and executed in parallel on CPU or GPU devices for high-performance computation.

The Taichi scope corresponds to the device side in CUDA.

## Template signature

Template signatures are what distinguish different instantiations of a kernel template.

For example, The signature of `add(x, 42)` is `(x, ti.i32)`, which is the same as that of `add(x, 1)`. Therefore, the latter can reuse the previously compiled binary. The signature of `add(y, 42)` is `(y, ti.i32)`, a different value from the previous signature, hence a new kernel will be instantiated and compiled.

## Thread  local storage (TLS)

<https://en.wikipedia.org/wiki/Thread-local_storage>

## Traceback

<https://en.wikipedia.org/wiki/Stack_trace>

## Vertex shader

<https://en.wikipedia.org/wiki/Shader#Vertex_shaders>
