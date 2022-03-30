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
+- node
|
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
+- node
|
+------------+------------+------------+------------+------------+------------+------------+------------+
|  c0-lock   |  c1-lock   |  c2-lock   |  c3-lock   |  nullptr   |  *cell-1   |  *cell-2   |  nullptr   |
+------------+------------+------------+------------+------------+------------+------------+------------+
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

We can follow [`Pointer_activate`](https://github.com/taichi-dev/taichi/blob/0f4fb9c662e6e3ffacc26e7373258d8d0414423b/taichi/runtime/llvm/node_pointer.h#L41-L65) to see how `pointer` SNode is implemented using the sparse runtime infrastructure.

```cpp
void Pointer_activate(Ptr meta_, Ptr node, int i) {
  auto meta = (StructMeta *)meta_;
  auto num_elements = Pointer_get_num_elements(meta_, node);
  // 1
  volatile Ptr lock = node + 8 * i;
  volatile Ptr *data_ptr = (Ptr *)(node + 8 * (num_elements + i));
  // 2
  if (*data_ptr == nullptr) {
    // 3
    // The cuda_ calls will return 0 or do noop on CPUs
    u32 mask = cuda_active_mask();
    if (is_representative(mask, (u64)lock)) {
      // 4
      locked_task(
          lock,
          [&] {
            // 5
            auto rt = meta->context->runtime;
            auto alloc = rt->node_allocators[meta->snode_id];
            auto allocated = (u64)alloc->allocate();
            atomic_exchange_u64((u64 *)data_ptr, allocated);
          },
          [&]() { return *data_ptr == nullptr; });
    }
    warp_barrier(mask);
  }
}
```

1. Retrieves both the lock and the pointer for the `i`-th cell. Note that the pointer width is assumed to be 8-byte wide. Locks are simple 64-bit integers. According to the layout, there are `max_num_elements` number of locks, followed by `max_num_elements` number of pointer to cells.
2. Checks whether the content of `data_ptr` is `nullptr` without any lock. This is the classical [double-checked locking](https://en.wikipedia.org/wiki/Double-checked_locking) pattern.
3. If 2 is true, pick one thread within a CUDA warp to acquire the lock. This is a small optimization to prevent lock contention.
4. The winning thread tries to acquire the lock using [`locked_task`](https://github.com/taichi-dev/taichi/blob/master/taichi/runtime/llvm/locked_task.h).
5. Retrieves the memory allocator (`node_allocators`) for this particular SNode, allocates a new cell, and stores the address of the allocated cell into `data_pointer`. Because the cell size of each SNode is different, the runtime has a dedicated allocator for each SNode, which knows how much space to allocate per cell.

The deactivation and the checking-for-active procedures are quite similar. We omit them for brevity.

## `dynamic` SNode

`dynamic` is a special kind of SNodes in a few things:

* It must be a 1-D, terminating SNode. By terminating, it means `dynamic` can only be followed by `place`-ing a Taichi field.
* The axis of `dynamic` must be different from those of all its predecessors. For example:
  * `dense(ti.ij, (2, 4)).dynammic(ti.k, 8)`: This is OK, because `dynamic`'s axis is unique.
  * `dense(ti.ij, (2, 4)).dynammic(ti.j, 8)`: This results in an error, because `dynamic`'s axis and `dense`'s overlaps on axis `j`.
* `dynamic` can only store 32-bit integers. However, this is not an API contract, and is subject to change.

Below shows the layout of a `dynamic` SNode. Logically speaking, `dynamic` SNode can be viewed as `std::vector<int32_t>`. However, `dynamic` is implemented as a single linked list of *chunks*.

```sh
+- node
|
+------------+------------+------------+
|    lock    |     n      |    ptr     |
+------------+------------+------------+
                          |
                          +-+------------+  # chunk-0, chunk_start = 0
                            |     x------|--+
                            +------------+  |
                            |     0      |  |
                            +------------+  |
                            |     1      |  |
                            +------------+  |
                            |     2      |  |
                            +------------+  |
                            |     3      |  |
                            +------------+  |
                                            |
                                            +-+------------+  # chunk-1, chunk_start = 4
                                              |   nullptr  |
                                              +------------+
                                              |     4      |
                                              +------------+
                                              |     5      |
                                              +------------+
                                              |     6      |
                                              +------------+
                                              |     7      |
                                              +------------+
```

The activation/deacctivation process is no different from that of `pointer`. We can trace through [`Dynamic_append`](https://github.com/taichi-dev/taichi/blob/0f4fb9c662e6e3ffacc26e7373258d8d0414423b/taichi/runtime/llvm/node_dynamic.h#L61-L87) to better understand the layout.

```cpp
i32 Dynamic_append(Ptr meta_, Ptr node_, i32 data) {
  auto meta = (DynamicMeta *)(meta_);
  auto node = (DynamicNode *)(node_);
  auto chunk_size = meta->chunk_size;
  // 1
  auto i = atomic_add_i32(&node->n, 1);
  // 2
  int chunk_start = 0;
  auto p_chunk_ptr = &node->ptr;
  while (true) {
    // 3
    if (*p_chunk_ptr == nullptr) {
      locked_task(Ptr(&node->lock), [&] {
        if (*p_chunk_ptr == nullptr) {
          auto rt = meta->context->runtime;
          auto alloc = rt->node_allocators[meta->snode_id];
          *p_chunk_ptr = alloc->allocate();
        }
      });
    }
    // 4
    if (i < chunk_start + chunk_size) {
      // 4-1
      *(i32 *)(*p_chunk_ptr + sizeof(Ptr) +
               (i - chunk_start) * meta->element_size) = data;
      break;
    }
    // 4-2
    p_chunk_ptr = (Ptr *)(*p_chunk_ptr);
    chunk_start += chunk_size;
  }
  return i;
}
```

1. Uses the current length `n` as the index (`i`) to store `data`. 
2. `chunk_strat` tracks the starting index of a given chunk, and always starts at `0` . `p_chunk_ptr` is initialized to the pointer to the first chunk.
3. Inside the `while` loop, checks if the given chunk slot is empty first, and allocates a new chunk if so.
4. Compares if the determined index `i` falls within the current chunk.
   1. If so, stores `data` into the corresponding slot in this chunk. Note that the first `sizeof(Ptr)` bytes are skiped: they are reserved to store the address of the next chunk.
   2. Otherwise, jumps to the next chunk.


# Runtime

## Memory Allocator



## GC
