# How is Taichi's LLVM Sparse Runtime Implemented?

Taichi's LLVM Sparse Runtime lives under the [`taichi/runtime/llvm`](https://github.com/taichi-dev/taichi/tree/master/taichi/runtime/llvm) directory. It includes one header file for every single SNode type, and a [runtime.cpp](https://github.com/taichi-dev/taichi/blob/master/taichi/runtime/llvm/runtime.cpp) source file.

# SNode

For a single SNode type `X`, it comes with a `XMeta `struct derived from [`StructMeta`](https://github.com/taichi-dev/taichi/blob/2cdc58078ecd2aef2cde608f07325108c5b3d5a5/taichi/runtime/llvm/runtime.cpp#L365-L383). `StructMeta` has the following properties:

* `snode_id`: SNode ID.
* `element_size`: Size (in bytes) of the cell
* `max_num_elements`: Maximum number of cells this SNode can contain. For example, if you define a dense SNode `dense(ti.ij, shape=(2, 4))`, then `max_num_elements = 2 x 4 = 8`.

Note that "element" is a legacy name. The official terminology is a "cell".

SNode `X` implements the follwing APIs:

* `i32 X_get_num_elements(Ptr meta, Ptr node)`: Returns the capacity this SNode can hold. Note that it is *not* the number of active cells.
* `void X_active(Ptr meta, Ptr node, int i)`: Activates cell `i`.
* `i32 X_is_active(Ptr meta, Ptr node, int i)`: Returns if cell `i` is active.
* `Ptr X_lookup_element(Ptr meta, Ptr node, int i)`: Returns the pointer to cell `i`. This can be `nullptr` for sparse SNodes.

Here, `Ptr` is an alias for `uint8_t*`, and `i32` for `int32_t`. As the name suggests, `meta` points the corresponding meta struct, while `node` points to the SNode instance.

For sparse SNodes, they also implement this additional API:

* `void X_deactivate(Ptr meta, Ptr node, int i)`: Deactivates cell `i`.

However, this is likely subject to change, so that all SNodes can share the same set of APIs.

## `dense` SNode

`dense` is the simplest form of SNode. It is just an array of cells living in a chunk of contiguous memory, or `std::array<Cell, N>` for those with a C++ background. Its header file is in [`node_dense.h`](https://github.com/taichi-dev/taichi/blob/master/taichi/runtime/llvm/node_dense.h).


* `Dense_get_num_elements`: This is ust `max_num_elements` stored in `DenseMeta`.
* `Dense_activate`: Empty body, because cells are always activated in `dense`.
* `Dense_is_active`: Always returns `1`.
* `Dense_lookup_element`: Returns the address of the `i`-th cell. That is, `node + element_size * i`.

Layout of a `dense` SNode:

```sh
+------------+------------+------------+------------+
|            |            |            |            |
|   cell-0   |   cell-1   |   cell-2   |   cell-3   |
|            |            |            |            |
+------------+------------+------------+------------+
```

## `pointer` SNode

`pointer` is usually the go-to sparse SNode you should consider. It allocates memory only when a cell is actually activated, and recycles to a memory pool once it is deactivated. This saves the memory resource in large-scale grid computation. You can mentally view it as an `std::array<Cell*, N>`.

Layout of a `pointer` SNode:

```sh
+------------+------------+------------+------------+
|  nullptr   |  *cell-1   |  *cell-2   |  nullptr   |
+------------+------------+------------+------------+
             |            |
             |            +> +------------+
             |               |            |
             |               |   cell-2   |
             |               |            |
             |               +------------+
             |
             +-------------> +------------+
                             |            |
                             |   cell-1   |
                             |            |
                             +------------+
```



# Runtime

## Memory Allocator



## GC
