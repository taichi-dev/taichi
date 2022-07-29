# How is Taichi's LLVM Sparse Runtime Implemented?

Last update: 2022-04-21

---

Taichi's LLVM sparse runtime lives under the [`taichi/runtime/llvm`](https://github.com/taichi-dev/taichi/tree/master/taichi/runtime/llvm) directory. It includes one header file for every single SNode type, and a [runtime.cpp](https://github.com/taichi-dev/taichi/blob/master/taichi/runtime/llvm/runtime.cpp) source file.

# SNode

For a single SNode type `X` (where `X` can be `dense`, `bitmasked`, `pointer` or `dynamic`), it comes with a `XMeta` struct derived from [`StructMeta`](https://github.com/taichi-dev/taichi/blob/2cdc58078ecd2aef2cde608f07325108c5b3d5a5/taichi/runtime/llvm/runtime.cpp#L365-L383). `StructMeta` has the following properties:

* `snode_id`: SNode ID.
* `i32 X_get_num_elements(Ptr meta, Ptr node)`: Returns the capacity this SNode can hold. Note that it is *not* the current number of active cells, but the maximum.
* `void X_activate(Ptr meta, Ptr node, int i)`: Activates cell `i`.
* `i32 X_is_active(Ptr meta, Ptr node, int i)`: Returns if cell `i` is active.
* `Ptr X_lookup_element(Ptr meta, Ptr node, int i)`: Returns the pointer to cell `i`. This can be `nullptr` for sparse SNodes.

Here, `Ptr` is an alias for `uint8_t*`, and `i32` for `int32_t`. As the name suggests, `meta` points to the corresponding meta struct, while `node` points to the SNode instance.

For sparse SNodes, they also implement this additional API:

* `void X_deactivate(Ptr meta, Ptr node, int i)`: Deactivates cell `i`.

However, this is likely subject to change, so that all SNodes can share the same set of APIs.

## `dense` SNode

`dense` is the simplest form of SNode. It is just an array of cells living in a chunk of contiguous memory, or `std::array<Cell, N>` for those with a C++ background. Its header file is in [`node_dense.h`](https://github.com/taichi-dev/taichi/blob/master/taichi/runtime/llvm/node_dense.h).


* `Dense_get_num_elements`: This is just `max_num_elements` stored in `DenseMeta`.
* `Dense_activate`: No-op, because cells are always activated in `dense`.
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

Upon initialization, Taichi preallocates a chunk of memory space, namely `ambient_elements`, which is shared across all the inactive sparse SNodes (`dynamic` and `pointer`). Therefore dereferencing an inactive sparse SNode will generate the default value stored in `ambient_elements` (usually zero).

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

We can follow [`Pointer_activate`](https://github.com/taichi-dev/taichi/blob/0f4fb9c662e6e3ffacc26e7373258d8d0414423b/taichi/runtime/llvm/node_pointer.h#L41-L65) to see how the `pointer` SNode is implemented using the sparse runtime infrastructure.

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

1. Retrieves both the lock and the pointer for the `i`-th cell. Note that the pointer width is assumed to be 8-byte wide. Locks are simply 64-bit integers. According to the layout, there are `max_num_elements` number of locks, followed by `max_num_elements` number of pointers to cells.
2. Checks whether the content of `data_ptr` is `nullptr` without any locking. This is the classical [double-checked locking](https://en.wikipedia.org/wiki/Double-checked_locking) pattern.
3. If 2 is true, pick one thread within a CUDA warp to acquire the lock. This is a small optimization to prevent lock contention. On pre-Volta devices without [independent thread scheduling](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html#sm-independent-thread-scheduling), this also prevents deadlocking.
4. The winning thread tries to acquire the lock using [`locked_task`](https://github.com/taichi-dev/taichi/blob/master/taichi/runtime/llvm/locked_task.h).
5. Retrieves the memory allocator (`node_allocators`) for this particular SNode, allocates a new cell, and stores the address of the allocated cell into `data_ptr`. Because the cell size of each SNode is different, the runtime has a dedicated allocator for each SNode, which knows how much space to allocate per cell.

The deactivation and the checking-for-active procedures are quite similar. We omit them for brevity.

## `dynamic` SNode

`dynamic` is a special kind of SNodes in a few ways:

* It must be a 1-D, terminating SNode. By terminating, it means `dynamic` can only be followed by `place`-ing a Taichi field.
* The axis of `dynamic` must be different from those of all its predecessors. For example:
  * `dense(ti.ij, (2, 4)).dynamic(ti.k, 8)`: This is OK, because `dynamic`'s axis is unique.
  * `dense(ti.ij, (2, 4)).dynamic(ti.j, 8)`: This results in an error, because `dynamic`'s axis and `dense`'s overlaps on axis `j`.

Below shows the layout of a `dynamic` SNode. Logically speaking, `dynamic` SNode can be viewed as `std::vector<int32_t>`. However, `dynamic` is implemented as a singly linked list of *chunks*.

```sh
# `n` stores the number of elements in this dynammic SNode cell

+- node
|
+------------+------------+------------+
|    lock    |     n      |    ptr     |
+------------+------------+------------+
                          |
                          +-+------------+  # chunk-0, chunk_start = 0
                            |     >------|--+
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

The activation/deactivation process is no different from that of `pointer`. We can trace through [`Dynamic_append`](https://github.com/taichi-dev/taichi/blob/0f4fb9c662e6e3ffacc26e7373258d8d0414423b/taichi/runtime/llvm/node_dynamic.h#L61-L87) to better understand the layout.

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
   1. If so, stores `data` into the corresponding slot in this chunk. Note that the first `sizeof(Ptr)` bytes are skipped: they are reserved to store the address of the next chunk.
   2. Otherwise, jumps to the next chunk.

# Runtime

Runtime for the LLVM backends is in https://github.com/taichi-dev/taichi/blob/master/taichi/runtime/llvm/runtime.cpp. Note that this file is *NOT* linked into Taichi's core C++ library. Instead, it is compiled into a LLVM byte code file (`.bc`). Upon starting Taichi, the `.bc` file is loaded back into the memory, de-serialized into an `llvm::Module`, and linked together with the JIT compiled Taichi kernels. This design has the advantage that the runtime code can be written once and shared between backends using LLVM, such as CPU and CUDA. In addition, the sparse runtime can be implemented in a language with enough abstraction (i.e., C++ instead of raw LLVM IR).

The core data structure of this runtime is [`LLVMRuntime`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L543), which holds a handful of data:

* All the root SNodes info (`roots` and `root_mem_sizes`)
* SNode memory allocators
* Random states for supporting `ti.random` (`rand_states`)
* Print and error message buffers
* ...

We will explain how the SNode memory allocator is implemented, which is the bedrock of the sparse SNodes.

## `NodeManager`: a recycling memory allocator

Each SNode is associated with its own memory allocator. These allocators are stored in an array, [`node_allocators`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L562).

The allocator is of type [`NodeManager`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L619). It contains [three linked lists of type `ListManager`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L627):

* `data_list`: A list of fixed-sized memory chunks. Each chunk can store `chunk_num_elements` SNode cells. The aforementioned SNode chunks are from this list.
* `free_list`: Indices of the free SNode cells. Each node in this list is an `int32_t` (or [`list_data_type`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L630)). When allocating, the runtime will first try to reuse a cell in the free list if there is one available, before requesting extra space from the memory allocator. (More details below.)
* `recycled_list`: Indices of the released SNode cells. After a GC execution, items in this list will be transferred into `free_list` for reuse. Each node in this list is also an `int32_t`.

Here's how [`allocate()`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L655) is implemented:

```cpp
Ptr allocate() {
  // 1
  int old_cursor = atomic_add_i32(&free_list_used, 1);
  i32 l;
  if (old_cursor >= free_list->size()) {
    // 2
    l = data_list->reserve_new_element();
  } else {
    // 3
    l = free_list->get<list_data_type>(old_cursor);
  }
  // 4
  return data_list->get_element_ptr(l);
}
```

1. Reads the (possible) index of the next free item.
2. If running out of the indices in the free list, allocate a new chunk from `data_list`.
3. Otherwise, re-use the index from `free_list`.
4. Either way, index `l` points to a memory slot in `data_list`. Returns that slot.

[`recycle()`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L672-L675) is quite straightforward.

Before jumping into the garbage collection (GC) system, we will also take look at the lower-level `ListManager`.

## `ListManager`: a CPU/GPU linked list

The way [`ListManager`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L416-L423) gets implemented is described as follows:

> A simple list data structure that is infinitely long. Data are organized in chunks, where each chunk is allocated on demand.

Calling it a linked list can be a bit misleading. In fact, it is more like `std::deque` in C++. `ListManager` holds an array of memory chunks in `chunks`, whereas each chunk can hold `max_num_elements_per_chunk` elements.

[`ListManager`'s constructor](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L433-L444) shows the necessary member variables:

```cpp
ListManager(LLVMRuntime *runtime,
            std::size_t element_size,
            std::size_t num_elements_per_chunk)
    : element_size(element_size),
      max_num_elements_per_chunk(num_elements_per_chunk),
      runtime(runtime) {
  taichi_assert_runtime(runtime, is_power_of_two(max_num_elements_per_chunk),
                        "max_num_elements_per_chunk must be POT.");
  lock = 0;
  num_elements = 0;
  log2chunk_num_elements = taichi::log2int(num_elements_per_chunk);
}
```

The important method for allocating a new element is [`reserve_new_element()`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L448-L453):

```cpp
i32 reserve_new_element() {
  auto i = atomic_add_i32(&num_elements, 1);
  auto chunk_id = i >> log2chunk_num_elements;
  touch_chunk(chunk_id);
  return i;
}
```

It increments `num_elements` to get the index of this new element, and calculates the belonging chunk ID. Then it ensures that the chunk is actually allocated using [`touch_chunk()`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L1570-L1584):

```cpp
void ListManager::touch_chunk(int chunk_id) {
  taichi_assert_runtime(runtime, chunk_id < max_num_chunks,
                        "List manager out of chunks.");
  if (!chunks[chunk_id]) {
    locked_task(&lock, [&] {
      // may have been allocated during lock contention
      if (!chunks[chunk_id]) {
        grid_memfence();
        auto chunk_ptr = runtime->request_allocate_aligned(
            max_num_elements_per_chunk * element_size, 4096);
        atomic_exchange_u64((u64 *)&chunks[chunk_id], (u64)chunk_ptr);
      }
    });
  }
}
```

By now, you should be quite familiar with this double-checked locking pattern. One thing noteworthy is that the fundamental memory allocation function is provided by [`LLVMRuntime::request_allocate_aligned()`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L846-L866).

[`get_element_ptr()`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L480-L483) illustrates the way to lookup the address of an element from a given index:

```cpp
Ptr get_element_ptr(i32 i) {
  return chunks[i >> log2chunk_num_elements] +                       // chunk base
          element_size * (i & ((1 << log2chunk_num_elements) - 1));  // slot within the chunk
}
```

There is also a reverse operation, [`ptr2index()`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L499-L509). It iterates over each chunk to see if the passed-in `ptr` falls into that memory range.

With these primitive helpers, [`allocate()`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L1591-L1594) is trivial to implement:

```cpp
Ptr ListManager::allocate() {
  auto i = reserve_new_element();
  return get_element_ptr(i);
}
```

So is [`append()`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L1586-L1589):

```cpp
void ListManager::append(void *data_ptr) {
  auto ptr = allocate();
  std::memcpy(ptr, data_ptr, element_size);
}
```

[`clear()`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L472-L474) simply resets `num_elements` to zero, without doing anything to the list contents.

## Garbage collection (GC)

GC will happen (in a parallel way on GPUs), after an offloaded task with possible sparse SNode deactivations.
The GC process for a given SNode is divided into three stages:

1. [`gc_parallel_0`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L1600-L1626): Moves the remaining, unused indices in `free_list` to its head.

Normally this can be done via a simple `for` loop that copies data. However, we need to run it in parallel on GPUs and the data source and destination may overlap, so special care is needed.
Therefore, the code differentiates between the cases where destination and source ranges overlap or not:

```sh
# Legend: "\\\" for already re-used cell indices, "   " (empty) for cell indices that are still available.
# src and dst range do not overlap:  src=[6, 8), dst=[0, 2)
0   1   2   3   4   5   6   7   8        0   1   2
+---+---+---+---+---+---+---+---+        +---+---+
|\\\|\\\|\\\|\\\|\\\|\\\|   |   |  --->  |   |   |
+---+---+---+---+---+---+---+---+        +---+---+
                        src              dst

# src and dst range overlap: src=[3, 8), dst=[0, 5)
0   1   2   3   4   5   6   7   8        0   1   2   3   4   5
+---+---+---+---+---+---+---+---+        +---+---+---+---+---+
|\\\|\\\|\\\|   |   |   |   |   |  --->  |   |   |   |   |   |
+---+---+---+---+---+---+---+---+        +---+---+---+---+---+
            src                          dst
```

Note cell indices 3 and 4 are in both the source and the destination regions.

2. [`gc_parallel_1`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L1628-L1640): Does some bookkeeping for `free_list` and `recycled_list`.

```cpp
void gc_parallel_1(RuntimeContext *context, int snode_id) {
  LLVMRuntime *runtime = context->runtime;
  auto allocator = runtime->node_allocators[snode_id];
  auto free_list = allocator->free_list;

  const i32 num_unused =
      max_i32(free_list->size() - allocator->free_list_used, 0);
  free_list->resize(num_unused);

  allocator->free_list_used = 0;
  // 1
  allocator->recycle_list_size_backup = allocator->recycled_list->size();
  // 2
  allocator->recycled_list->clear();
}
```

It is important to understand that this stage must be run on a single thread. This is to avoid any kind of data race on `recycled_list`. During GC, `recycled_list` must be cleared, with all its contents being transferred into `free_list`. If this was done in the third stage (which is run in parallel), it is quite difficult to coordinate the event sequence of clearing and transferring among the GPU threads. As a result, this serial stage is created in order to store the number of elements of `recycled_list` into `recycle_list_size_backup` in advance (1), then clears the list (2). Fortunately, computation here is light and it doesn't take too much time even running in serial.

3. [`gc_parallel_2`](https://github.com/taichi-dev/taichi/blob/172cab8a57fcfc2d766fe2b7cd40af669dadf326/taichi/runtime/llvm/runtime.cpp#L1642-L1681): Replenishes `free_list` with the indices in `recycled_list`, then zero-fills all the recycled memory locations.

```cpp
void gc_parallel_2(RuntimeContext *context, int snode_id) {
  LLVMRuntime *runtime = context->runtime;
  auto allocator = runtime->node_allocators[snode_id];
  // 1
  auto elements = allocator->recycle_list_size_backup;
  auto free_list = allocator->free_list;
  auto recycled_list = allocator->recycled_list;
  auto data_list = allocator->data_list;
  auto element_size = allocator->element_size;
  using T = NodeManager::list_data_type;
  // 2
  auto i = block_idx();
  while (i < elements) {
    // 3
    auto idx = recycled_list->get<T>(i);
    auto ptr = data_list->get_element_ptr(idx);
    // 4
    if (thread_idx() == 0) {
      free_list->push_back(idx);
    }
    // memset
    // 5
    auto ptr_stop = ptr + element_size;
    if ((uint64)ptr % 4 != 0) {
      auto new_ptr = ptr + 4 - (uint64)ptr % 4;
      if (thread_idx() == 0) {
        for (uint8 *p = ptr; p < new_ptr; p++) {
          *p = 0;
        }
      }
      ptr = new_ptr;
    }
    // now ptr is a multiple of 4
    // 6
    ptr += thread_idx() * sizeof(uint32);
    while (ptr + sizeof(uint32) <= ptr_stop) {
      *(uint32 *)ptr = 0;
      ptr += sizeof(uint32) * block_dim();
    }
    while (ptr < ptr_stop) {
      *ptr = 0;
      ptr++;
    }
    // 7
    i += grid_dim();
  }
}
```

1. Reads back the size of `recycled_list` before clearing stored in stage 2.
2. `i` is set to the CUDA block index. Note that all threads within a CUDA block cooperatively zero-fills the memory region of a SNode cell.
3. Reads the SNode cell index stored in the `i`-th element in `recycled_list`. Recall that `ListManager::clear()` doesn't really modify its content, so it's still safe to read this list.
4. The first thread of the block gets to append this cell index into `free_list`.
5. This begins the zero-fill process. First thing first, thread 0 clears the first few bytes, in case the cell is not 4-byte aligned.
6. Each thread clears its corresponding 4 bytes, then jumps to the next location, with stride being `sizeof(uint32) * block_dim()`.

    ```sh
    # A SNode cell of 40 bytes, zero-filled by 4 threads:
    +----+----+----+----+----+----+----+----+----+----+
    | t0 | t1 | t2 | t3 | t0 | t1 | t2 | t3 | t0 | t1 |
    +----+----+----+----+----+----+----+----+----+----+
    ```

7. The entire thread block jumps to the next recycled SNode cell.
