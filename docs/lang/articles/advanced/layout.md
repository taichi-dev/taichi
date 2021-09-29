---
sidebar_position: 2
---

# Fields (advanced)

This section introduces some advanced features of Taichi fields.
Make sure you have gone through [Fields](../basic/field).

## Packed mode

By default, all non-power-of-two dimensions of a field are automatically
padded to a power of two. For instance, a field of shape `(18, 65)` will
have an internal shape of `(32, 128)`. Although the padding has many benefits
such as allowing fast and convenient bitwise operations for coordinate handling,
it will consume potentially much more memory than expected.

If you would like to reduce memory usage, you can use the optional packed
mode. In packed mode, no padding is applied such that a field does not have a
larger internal shape than the defined shape when some of its dimensions
are not powers of two. The downside is that the runtime performance will
regress slightly.

A switch named `packed` for `ti.init()` decides whether to use packed mode:

```python
ti.init()  # default: packed=False
a = ti.field(ti.i32, shape=(18, 65))  # padded to (32, 128)
```

```python
ti.init(packed=True)
a = ti.field(ti.i32, shape=(18, 65))  # no padding
```

## Advanced data layouts

Apart from shape and data type, you can also specify the data layout of a
field in a recursive manner. This may allow you to achieve better performance.

Normally, you don't have to worry about the performance nuances between
different layouts, and you can just use the default one (simply by specifying
`shape` when creating fields) as a start.

However, when a field gets large, a proper data layout may be critical to
performance, especially for memory-bound applications. A carefully designed
data layout has much better spatial locality, which will significantly
improve cache/TLB-hit rates and cache line utilization.

Taichi decouples computation from data structures, and the Taichi compiler
automatically optimizes data accesses on a specific data layout. This allows
you to quickly experiment with different data layouts and figure out the most
efficient one on a specific task and computer architecture.

### Layout 101: from `shape` to `ti.root.X`

The following declares a 0-D field:

```python {1-2}
x = ti.field(ti.f32)
ti.root.place(x)
# is equivalent to:
x = ti.field(ti.f32, shape=())
```

The following declares a 1D field of shape `3`:

```python {1-2}
x = ti.field(ti.f32)
ti.root.dense(ti.i, 3).place(x)
# is equivalent to:
x = ti.field(ti.f32, shape=3)
```

The following declares a 2D field of shape `(3, 4)`:

```python {1-2}
x = ti.field(ti.f32)
ti.root.dense(ti.ij, (3, 4)).place(x)
# is equivalent to:
x = ti.field(ti.f32, shape=(3, 4))
```

After being comfortable with these equivalent definitions, you can move forward
and see how to change the data layout.

### Row-major versus column-major

As you might have learned in a computer architecture course,
address spaces are linear in modern computers. To
simplify the discussions, data type size will not be considered and will always
be treated as 1. Assume the starting address of a field is `base`. Then for 1D
Taichi fields, the address of the `i`-th element is simply `base + i`.

However, a multi-dimensional field has to be flattened in order to fit into the
1D address space. For example, there are two ways to store a 2D field of size `(3, 2)`:

- Row-major: let the address of the `(i, j)`-th element be `base + i * 2 + j`;
- Column-major: let the address of the `(i, j)`-th element be
  `base + j * 3 + i`.

To specify which layout to use (default layout is row-major):

```python
ti.root.dense(ti.i, 3).dense(ti.j, 2).place(x)   # row-major
ti.root.dense(ti.j, 2).dense(ti.i, 3).place(y)   # column-major
```

Both `x` and `y` have shape `(3, 2)`, and they can be accessed in the same
manner with `x[i, j]` and `y[i, j]`, where `0 <= i < 3 && 0 <= j < 2`. However,
they have different memory layouts:

```
# address:  low ........................................... high
#       x:  x[0, 0]  x[0, 1]  x[1, 0]  x[1, 1]  x[2, 0]  x[2, 1]
#       y:  y[0, 0]  y[1, 0]  y[2, 0]  y[0, 1]  y[1, 1]  y[2, 1]
```

:::note

For those who are familiar with C/C++, here is what they look like in C code:

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

:::

### Array of Structures (AoS) versus Structure of Arrays (SoA)

Fields of same shape can be placed together.

For example, the following places two 1D fields of shape `3` together, which
is called Array of Structures (AoS):

```python
ti.root.dense(ti.i, 3).place(x, y)
```

Their memory layout is:

```
#  address: low ......................... high
#           x[0]  y[0]  x[1]  y[1]  x[2]  y[2]
```

By contrast, the following places these two fields separately, which is called
Structure of Arrays (SoA):

```python
ti.root.dense(ti.i, 3).place(x)
ti.root.dense(ti.i, 3).place(y)
```

Now their memory layout is:

```
#  address: low ......................... high
#           x[0]  x[1]  x[2]  y[0]  y[1]  y[2]
```

**To improve spatial locality of memory accesses, it may be helpful to
place data elements that are often accessed together within
relatively close addresses.** Take a simple 1D wave equation solver as an example:

```python
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

Here, `pos` and `vel` are placed separately, so the distance in address
space between `pos[i]` and `vel[i]` is `200000`. This results in poor spatial
locality and poor performance. A better way is to place them together:

```python
ti.root.dense(ti.i, N).place(pos, vel)
```

Then `vel[i]` is placed right next to `pos[i]`, which can increase spatial
locality and therefore improve performance.

### Flat layouts versus hierarchical layouts

From the above discussions,

```python
val = ti.field(ti.f32, shape=(32, 64, 128))
```

is equivalent to the following in C/C++:

```c
float val[32][64][128];
```

However, at times this data layout may be suboptimal for certain types of
computation tasks. For example, in trilinear texture interpolation,
`val[i, j, k]` and `val[i + 1, j, k]` are often accessed together. With the
above layout, they are very far away (32 KB) from each other, and not even
within the same 4 KB pages. This creates a huge cache/TLB pressure and leads
to poor performance.

A better layout might be

```python
val = ti.field(ti.f32)
ti.root.dense(ti.ijk, (8, 16, 32)).dense(ti.ijk, (4, 4, 4)).place(val)
```

This organizes `val` in `4x4x4` blocks, so that with high probability
`val[i, j, k]` and its neighbours are close to each other (i.e., in the
same cache line or memory page).

### Struct-fors on advanced dense data layouts

Struct-fors on nested dense data structures will automatically follow their
layout in memory. For example, if 2D scalar field `A` is defined in row-major,

```python
for i, j in A:
    A[i, j] += 1
```

will iterate over elements of `A` following the row-major order. Similarly, if
`A` is defined in column-major, then the iteration follows the column-major
order.

If `A` is hierarchical, it will be iterated level by level. This maximizes the
memory bandwidth utilization in most cases.

As you may notice, only dense data layouts are covered in this section. For sparse
data layouts, see [Sparse computation](./sparse.md).

### More examples of advanced dense data layouts

2D field, row-major:

```python
A = ti.field(ti.f32)
ti.root.dense(ti.ij, (256, 256)).place(A)
```

2D field, column-major:
```python
A = ti.field(ti.f32)
ti.root.dense(ti.j, 256).dense(ti.i, 256).place(A)
```

_8x8_-blocked 2D field of size _1024x1024_:

```python
density = ti.field(ti.f32)
ti.root.dense(ti.ij, (128, 128)).dense(ti.ij, (8, 8)).place(density)
```

3D particle positions and velocities, AoS:

```python
pos = ti.Vector.field(3, dtype=ti.f32)
vel = ti.Vector.field(3, dtype=ti.f32)
ti.root.dense(ti.i, 1024).place(pos, vel)
# equivalent to
ti.root.dense(ti.i, 1024).place(pos.get_scalar_field(0),
                                pos.get_scalar_field(1),
                                pos.get_scalar_field(2),
                                vel.get_scalar_field(0),
                                vel.get_scalar_field(1),
                                vel.get_scalar_field(2))
```

3D particle positions and velocities, SoA:

```python
pos = ti.Vector.field(3, dtype=ti.f32)
vel = ti.Vector.field(3, dtype=ti.f32)
for i in range(3):
    ti.root.dense(ti.i, 1024).place(pos.get_scalar_field(i))
for i in range(3):
    ti.root.dense(ti.i, 1024).place(vel.get_scalar_field(i))
```

## Dynamic field allocation and destruction

You can use the `FieldsBuilder` class for dynamic field allocation and destruction.
`FieldsBuilder` has the same data structure declaration APIs as `ti.root`,
including `dense()`. After declaration, you need to call the `finalize()`
method to compile it to an `SNodeTree` object.

A simple example is:

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

fb2 = ti.FieldsBuilder()
y = ti.field(dtype=ti.f32)
fb2.dense(ti.i, 5).place(y)
fb2_snode_tree = fb2.finalize()  # Finalizes the FieldsBuilder and returns a SNodeTree
func(y)
```

In fact, `ti.root` is implemented by `FieldsBuilder` implicitly, so you can
allocate the fields directly under `ti.root`:
```python
import taichi as ti
ti.init()  # Implicitly: ti.root = ti.FieldsBuilder()

@ti.kernel
def func(v: ti.template()):
    for I in ti.grouped(v):
        v[I] += 1

x = ti.field(dtype=ti.f32)
ti.root.dense(ti.ij, (5, 5)).place(x)
func(x)  # Automatically calls ti.root.finalize()
# Implicitly: ti.root = ti.FieldsBuilder()

y = ti.field(dtype=ti.f32)
ti.root.dense(ti.i, 5).place(y)
func(y)  # Automatically calls ti.root.finalize()
```

Furthermore, if you don't want to use the fields under a certain `SNodeTree`
anymore, you could call the `destroy()` method on the finalized `SNodeTree`
object, which will recycle its memory into the memory pool:

```py
import taichi as ti
ti.init()

@ti.kernel
def func(v: ti.template()):
    for I in ti.grouped(v):
        v[I] += 1

fb = ti.FieldsBuilder()
x = ti.field(dtype=ti.f32)
fb.dense(ti.ij, (5, 5)).place(x)
fb_snode_tree = fb.finalize()  # Finalizes the FieldsBuilder and returns a SNodeTree
func(x)

fb_snode_tree.destroy()  # x cannot be used anymore
```
