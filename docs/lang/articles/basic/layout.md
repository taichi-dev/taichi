---
sidebar_position: 2
---

# Fields (advanced)

Modern processor cores compute orders of magnitude faster than their equipped memory systems. To shrink this  performance gap, multi-level cache systems and high-bandwidth multi-channel memories are built into the computer architectures.

After familiarizing yourself with the basics of Taichi [Fields](./field.md), this article helps you one step further by explaining the underlying memory layout that is essential to write high-performance Taichi programs. In particular, we present how to organize an efficient data layout and how to manage memory occupancy.

## Organize an efficient data layout

In this section, we introduce how to organize data layouts in Taichi fields. The central principle of efficient data layout is _locality_. Generally speaking, a program with desirable locality has at least one of the following features:

* Dense data structure
* Loop over data in small-range (within 32KB is good for most processors)
* Sequential load/store

:::note


Be aware that data are always fetched from memory in blocks (pages). The hardware has little knowledge about how a specific data element is used in the block. The processor blindly fetch the entire block according to the requested memory address. Therefore, the memory bandwidth is wasted when data are not fully utilized.

For sparse fields, see the [Sparse computation](./sparse.md).

:::

### Layout 101: from `shape` to `ti.root.X`

<!-- haidong: what's else optional in ti.root? -->
In basic usages, we use the `shape` descriptor to construct a field. Taichi provides flexible statements to describe more advanced data organizations, the `ti.root.X`.
Let's get some familiarity with examples:

* Declare a 0-D field:

```python {1-2}
x = ti.field(ti.f32)
ti.root.place(x)
# is equivalent to:
x = ti.field(ti.f32, shape=())
```

* Declare a 1-D field of shape `3`:

```python {1-2}
x = ti.field(ti.f32)
ti.root.dense(ti.i, 3).place(x)
# is equivalent to:
x = ti.field(ti.f32, shape=3)
```

* Declare a 2-D field of shape `(3, 4)`:

```python {1-2}
x = ti.field(ti.f32)
ti.root.dense(ti.ij, (3, 4)).place(x)
# is equivalent to:
x = ti.field(ti.f32, shape=(3, 4))
```

You can also nest two 1D `dense` statements to describe the same 2D array.

```python {1-2}
x = ti.field(ti.f32)
ti.root.dense(ti.i, 3).dense(ti.j, 4).place(x)
```

In a nutshell, the `ti.root.X` statement progressively binds a shape to the corresponding axis.
By nesting multiple statements, we can construct a field with higher dimensions.
<!-- haidong: how far can we go? how many default axis exist? -->

In order to traverse the nested statements, you can use `struct-for`:

```python {1}
for i, j in A:
    A[i, j] += 1
```

The order to access `A`, namely the order to iterate `i` and `j`, affects the program performance subtly. The Taichi compiler is capable of automatically deducing the underlying data layout and applying a proper access order. This is an advantage over most general-purpose programming languages where the access order has to be optimized manually.

### Row-major versus column-major

Memory address space is linear as you may have learned from a computer architecture course. Without loss of generality, we omit the differences in data types and assume each data element has size 1. Moreover, we denote the starting memory address of a field as `base`, and the indexing formula for 1D Taichi fields is `base + i` for the `i`-th element.

For multi-dimensional fields, we can flatten the high-dimension index into the linear memory address space in two ways: Taking a 2D field of shape `(M, N)` as an instance, we can either store `M` rows with `N`-length 1D buffers, say the _row-major_ way, or store `N` columns, say the _column-major_ way. The index flatten formula for the `(i, j)`-th element is `base + i * N + j` for row-major and `base + j * M + i` for column-major, respectively.

We can easily derive that elements in the same row are close in memory for row-major fields. The selection of the optimal layout is based on how the elements are accessed, namely, the access patterns. Patterns such as frequently accessing elements of the same row in a column-major field typically lead to performance degradation.

The default Taichi field layout is row-major. With the `ti.root` statements, fields can be defined as follows:

```python
ti.root.dense(ti.i, M).dense(ti.j, N).place(x)   # row-major
ti.root.dense(ti.j, N).dense(ti.i, M).place(y)   # column-major
```

In the code above, the axis denotation in the rightmost `dense` statement indicates the continuous axis. For the `x` field, elements in the same row (with same `i` and different `j`) are close in memory, hence it's row-major; For the `y` field, elements in the same column (same `j` and different `i`) are close, hence it's column-major. With an example of (2, 3), we visualize the memory layouts of `x` and `y` as follows:

```
# address:  low ........................................... high
#       x:  x[0, 0]  x[0, 1]  x[1, 0]  x[1, 1]  x[2, 0]  x[2, 1]
#       y:  y[0, 0]  y[1, 0]  y[2, 0]  y[0, 1]  y[1, 1]  y[2, 1]
```

It is worth noting that the accessor is unified for Taichi fields: the `(i, j)`-th element in the field is accessed with the identical 2D index `x[i, j]` and `y[i, j]`. Taichi handles the layout variants and applies proper indexing equations internally. Thanks to this feature, users can specify their desired layout at definition, and use the fields without concerning about the underlying memory organizations. To change the layout, it's sufficient to just swap the order of `dense` statements, and leave rest of the code intact.

:::note

For readers who are familiar with C/C++, below is an example C code snippet that demonstrates data access in 2D arrays:

```c
int x[3][2];  // row-major
int y[2][3];  // column-major

for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
        do_something(x[i][j]);
        do_something(y[j][i]);
    }
}
```

The accessors of `x` and `y` are in reverse order between row-major arrays and column-major arrays, respectively. Compared with Taichi fields, there is much more code to revise when you change the memory layout.

:::

<!-- ### Array of Structures (AoS) versus Structure of Arrays (SoA) -->

### AoS versus SoA

AoS means _array of structures_ and SoA means _structure of arrays_. Consider an RGB image with 4 pixels and 3 color channels, an AoS layout stores `RGBRGBRGBRGB` while an SoA layout stores `RRRRGGGGBBBB`.

The selection of AoS or SoA layout largely depends on the access pattern to the field. Let's discuss a scenario to process large RGB images. The two layouts have the following arrangements in memory:

```
# address: low ...................... high
# AoS:     RGBRGBRGBRGBRGBRGB.............
# SoA:     RRRRR...RGGGGGGG...GBBBBBBB...B
```

To calculate grey scale of each pixel, you need all color channels but do not require the value of other pixels. In this case, the AoS layout has a better memory access pattern: Since color channels are stored continuously, and adjacent channels can be fetched instantly. The SoA layout is not a good option because the color channels of a pixel are stored far apart in the memory space.

We describe how to construct AoS and SoA fields with our `ti.root.X` statements. The SoA fields are trivial:

```python
x = ti.field(ti.f32)
y = ti.field(ti.f32)
ti.root.dense(ti.i, M).place(x)
ti.root.dense(ti.i, M).place(y)
```

where M is the length of `x` and `y`.
The data elements in `x` and `y` are continuous in memory:

```
#  address: low ................................. high
#           x[0]  x[1]  x[2] ... y[0]  y[1]  y[2] ...
```

For AoS fields, we construct the field with

```python
x = ti.field(ti.f32)
y = ti.field(ti.f32)
ti.root.dense(ti.i, M).place(x, y)
```

The memory layout then becomes

```
#  address: low .............................. high
#           x[0]  y[0]  x[1]  y[1]  x[2]  y[2] ...
```

Here, `place` interleaves the elements of Taichi fields `x` and `y`.

As previously introduced, the access methods to `x` and `y` remain the same for both  AoS and SoA. Therefore, the data layout can be changed flexibly without revising the application logic.

<!-- haidong: I hope this part is 1) revised to a runnable and complete example 2) provides performance constrast-->
For better illustration, let's see an example of an 1D wave equation solver:

```python
N = 200000
pos = ti.field(ti.f32)
vel = ti.field(ti.f32)
# SoA placement
ti.root.dense(ti.i, N).place(pos)
ti.root.dense(ti.i, N).place(vel)

@ti.kernel
def step():
    pos[i] += vel[i] * dt
    vel[i] += -k * pos[i] * dt
```

The above code snippet defines SoA fields and a `step` kernel that sequentially accesses each element.
The kernel fetches an element from `pos` and `vel` for every iteration, respectively.
For SoA fields, the closest distance of any two elements in memory is `N`, which is unlikely to be efficient for large `N`.

We hereby switch the layout to AoS as follows:

```python
N = 200000
pos = ti.field(ti.f32)
vel = ti.field(ti.f32)
# AoS placement
ti.root.dense(ti.i, N).place(pos, vel)

@ti.kernel
def step():
    pos[i] += vel[i] * dt
    vel[i] += -k * pos[i] * dt
```

Merely revising the place statement is sufficient to change the layout. With this optimization, the instant elements `pos[i]` and `vel[i]` are now adjacent in memory, which is more efficient.

<!-- ```python
# SoA version
N = 200000
pos = ti.field(ti.f32)
vel = ti.field(ti.f32)
ti.root.dense(ti.i, N).place(pos)
ti.root.dense(ti.i, N).place(vel)

@ti.kernel
def step():
    pos[i] += vel[i] * dt
    vel[i] += -k * pos[i] * dt
```

```python
# AoS version
N = 200000
pos = ti.field(ti.f32)
vel = ti.field(ti.f32)
ti.root.dense(ti.i, N).place(pos)
ti.root.dense(ti.i, N).place(vel)

@ti.kernel
def step():
    pos[i] += vel[i] * dt
    vel[i] += -k * pos[i] * dt
```

Here, `pos` and `vel` for SoA are placed separately, so the distance in address
space between `pos[i]` and `vel[i]` is `200000`. This results in poor spatial
locality and poor performance. A better way is to place them together:

```python
ti.root.dense(ti.i, N).place(pos, vel)
```

Then `vel[i]` is placed right next to `pos[i]`, which can increase spatial
locality and therefore improve performance. -->

<!-- For example, the following places two 1D fields of shape `3` together, which
is called Array of Structures (AoS):

```python
ti.root.dense(ti.i, 3).place(x, y)
```

Their memory layout is:

By contrast, the following places these two fields separately, which is called
Structure of Arrays (SoA): -->

<!-- ```python
ti.root.dense(ti.i, 3).place(x)
ti.root.dense(ti.i, 3).place(y)
```
Now their memory layout is:


**To improve spatial locality of memory accesses, it may be helpful to
place data elements that are often accessed together within
relatively close addresses.**  -->


### AoS extension: hierarchical fields
<!-- haidong: I hope to remove this subsection. This content just repeats the AoS topic -->
Sometimes we want to access memory in a complex but fixed pattern, like traversing an image in 8x8 blocks. The apparent best practice is to flatten each 8x8 block and concatenate them together. From a Taichi user's perspective, however, the field is no longer a flat buffer. It now has a hierarchy with two levels: The image level and the block level. Equivalently, the field is an array of implicit 8x8 block structures.

We demonstrate the statements as follows:

```python
# Flat field
val = ti.field(ti.f32)
ti.root.dense(ti.ij, (M, N)).place(val)
```

```python
# Hierarchical field
val = ti.field(ti.f32)
ti.root.dense(ti.ij, (M // 8, N // 8)).dense(ti.ij, (8, 8)).place(val)
```

where `M` and `N` are multiples of 8. We encourage you to try this out! The performance difference can be significant!

## Manage memory occupancy

### Manual field allocation and destruction

Generally Taichi manages memory allocation and destruction without disturbing the users. However, there are times that users want explicit control over their memory allocations.

In this scenario, Taichi provides the `FieldsBuilder` for manual field memory allocation and destruction. `FieldsBuilder` features identical declaration APIs as `ti.root`. The extra step is to invoke `finalize()` at the end of all declarations. The `finalize()` returns an `SNodeTree` object to handle subsequent destructions.

Let's see a simple example:

```python
import taichi as ti
ti.init()

@ti.kernel
def func(v: ti.template()):
    for I in ti.grouped(v):
        v[I] += 1

fb1 = ti.FieldsBuilder()
x = ti.field(dtype=ti.f32)
fb1.dense(ti.ij, (5, 5)).place(x)
fb1_snode_tree = fb1.finalize()  # Finalizes the FieldsBuilder and returns a SNodeTree
func(x)
fb1_snode_tree.destroy()  # Destruction

fb2 = ti.FieldsBuilder()
y = ti.field(dtype=ti.f32)
fb2.dense(ti.i, 5).place(y)
fb2_snode_tree = fb2.finalize()  # Finalizes the FieldsBuilder and returns a SNodeTree
func(y)
fb2_snode_tree.destroy()  # Destruction
```

Actually, the above demonstrated `ti.root` statements are implemented with `FieldsBuilder`, despite that `ti.root` has the capability to automatically manage memory allocations and recycling.

### Packed mode

By default, Taichi implicitly fits a field in a larger buffer with power-of-two dimensions. We take the power-of-two padding convention because it is widely adopted in computer graphics. The design enables fast indexing with bitwise arithmetic and better memory address alignment, while trading off memory occupations.

For example, a `(18, 65)` field is materialized with a `(32, 128)` buffer, which is acceptable. As field size grows, the padding strategy can be exaggeratedly unbearable: `(129, 6553600)` will be expanded to `(256, 6335600)`, which allocates considerable unused blank memory. Therefore, Taichi provides the optional packed mode to allocate buffer that tightly fits the requested field shape. It is especially useful when memory usage is a major concern.

To leverage the packed mode, specify `packed` in `ti.init()` argument:

```python
ti.init()  # default: packed=False
a = ti.field(ti.i32, shape=(18, 65))  # padded to (32, 128)
```

```python
ti.init(packed=True)
a = ti.field(ti.i32, shape=(18, 65))  # no padding
```

You might observe mild performance regression with the packed mode due to more complex addressing and memory alignment. Therefore, the packed mode should be specified only when memory capacity is a major concern.
