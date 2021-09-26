---
sidebar_position: 3
---

# Sparse computation

Compiler-level support for spatially sparse computation is a unique feature of Taichi.

![image](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/sparse_grids.gif)

Figure: A swinging Taichi pattern represented with a 512x512 sparse grid. The sparse grid has a multi-level *tree* structure.
White stands for inactive tree nodes, and active tree nodes are darker.

The sparse grid above has the following structure:
- The grid is divided into 8x8 `block1` cells.
- Each `block1` cell has 4x4 `block2` sub-cells.
- Each `block2` cell has 4x4 `block3` sub-cells.
- Each `block3` cell has 4x4 sub-cells (pixels), each directly containing a `f32` value `x[i, j]`.

Taichi allows you to effortlessly define such data structure:

```python
x = ti.field(dtype=ti.i32)

block1 = ti.root.pointer(ti.ij, 8)
block2 = block1.pointer(ti.ij, 4)
block3 = block2.pointer(ti.ij, 4)
block3.dense(ti.ij, 4).place(x)
```
[[Full source code of this animation]](https://github.com/taichi-dev/taichi/blob/master/examples/features/sparse/taichi_sparse.py)

Intuitively, a sparse grid in Taichi allows you to use memory space more wisely, since only tree nodes involved in computation are allocated.
Now, let's take a step back and think about *why we need sparse grids, how to define them in Taichi, and how to compute on these data structures*.

## Motivation

High-resolution 2D/3D grids are often needed in large-scale spatial computation, such as physical simulation, rendering, and 3D reconstruction.
However, these grids tend to consume a huge amount of memory space and computation.
While a programmer may allocate large dense grids to store spatial data (especially physical quantities such as a density or velocity field),
oftentimes they only care about a small fraction of this dense grid, since the rest may be empty space (vacuum or air).

In short, the regions of interest in sparse grids may only occupy a small fraction of the whole bounding box.
If we can leverage such "spatial sparsity" and focus computation on the regions we care about,
we will significantly save storage and computing power.

:::note
The key to leverage spatial sparsity is to replace *dense* grids with *sparse* grids.
:::

On a sparse data structure, we say a pixel, voxel, or grid cell is *active*, if it is allocated and involved in computation.
The rest of the grid cells are simply *inactive*.
The *activity* of a leaf or intermediate cell is a boolean value. The activity value of a cell is `True` if and only if the cell is *active*.

Here is a simple example. The 2D multi-physics simulation (material point method) below has 256x256 grid cells.
Since the simulated objects do not fully occupy the whole domain, we would like to *adaptively* allocate the underlying simulation grid.
We subdivide the whole simulation domain into 16x16 *blocks*,
and each block has 16x16 *grid cells*.
Memory allocation can then happen at *block* granularity,
and we only consume memory space of blocks that are actually in the simulation.

![image](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi_elements/sparse_mpm_active_blocks.gif)

(Note the changing distribution of active blocks throughout the simulation.)

:::note
**Backend compatibility**: The LLVM backends (CPU/CUDA) and the Metal backend offer the full functionality of sparse computation.
Other backends offer no or limited support of sparse computation.
:::

:::note
Sparse matrices are usually **not** implemented in Taichi via (spatially-) sparse data structures. Use `ti.SparseMatrixBuilder` instead.
:::

## Defining sparse data structures in Taichi

Ideally, it would be nice to have a sparse voxel data structure that consumes space or computation only when the voxels are active.
Practically, Taichi programmers use hierarchical data structures (trees) to organize sparse voxel data.
To concentrate computation on sparse regions of interest, multi-level sparse voxel data structures are studied extensively.
These data structure are essentially trees.

### Data structure hierarchy

Traditionally, [Quadtrees](https://en.wikipedia.org/wiki/Quadtree) (2D) and
[Octrees](https://en.wikipedia.org/wiki/Octree) (3D) are often adopted.
Since dereferencing pointers is relatively costly on modern computer architectures,
compared to quadtrees and octrees, it is more performance-friendly to use trees with larger branching factors and
shallower structures.
[VDB](https://www.openvdb.org/) and [SPGrid](http://pages.cs.wisc.edu/~sifakis/papers/SPGrid.pdf) are such examples.
In Taichi, programmers can compose data structures similar to VDB and SPGrid with SNodes.

TODO: add an image here.

#### Blocked leaf cells and bitmasks

While a null pointer can effectively represent an empty sub-tree, at the leaf level using 64 bits to represent the activity
of a single voxel can consume too much space.
For example, if each voxel contains a single `f32` value (4 bytes), the 64-bit pointer pointing to the value itself would take 8 bytes.
In this case, storage costs of pointers triples the space to store the values, which goes against our goal to use sparse data structures to save space.

To amortize the storage cost of pointers, programmers usually organize voxels in a *blocked* manner,
and let the pointers directly point to the blocks (instead of voxels).

One caveat of such design is that voxels in the same `dense` block can no longer change its activity flexibly.
Instead, they share a single activity flag. To address this issue,
the `bitmasked` SNode additionally allocates 1-bit per voxel data to represent the voxel activity.

### A typical sparse data structure

```python
x = ti.field(dtype=ti.i32)

block = ti.root.pointer(ti.ij, (4, 4))
pixel = block.dense(ti.ij, (2, 2))
pixel.place(x)
```
## Computation on sparse data structures

### Activation on write

When writing to an inactive cell on a sparse data structure, Taichi automatically populates the data structure.

For example, when executing `x[4, 13] = 123` on the aforementioned sparse grid `x`,
Taichi automatically activates `block[1, 3]` so that `pixel[4, 13]` is allocated.

TODO: add image.

:::note
Reading an inactive voxel returns zero.
:::

### Sparse struct-fors

Efficiently looping over sparse grid cells that distribute irregularly can be a challenge, especially on parallel devices such as GPUs.
In Taichi, *struct-for's* natively support sparse data structures and only loops over voxels that are currently active.
The Taichi system ensures efficient parallelization.

```python
import taichi as ti

use_bitmask = True

ti.init()

x = ti.field(dtype=ti.i32)
block = ti.root.pointer(ti.ij, (4, 4))
if use_bitmask:
    pixel = block.bitmasked(ti.ij, (2, 2))
else:
    pixel = block.dense(ti.ij, (2, 2))
pixel.place(x)

@ti.kernel
def sparse_struct_for():
    x[2, 3] = 2
    x[5, 6] = 3

    for i, j in x:
        print('field x[{}, {}] = {}'.format(i, j, x[i, j]))

    for i, j in block:
        print('Active block: [{}, {}]'.format(i, j))

print('use_bitmask = {}'.format(use_bitmask))
sparse_struct_for()
```

You can loop over different levels of the cells. When `bitmask = True`, the program above generates
```
field x[2, 3] = 2
field x[5, 6] = 3
Active block: [1, 1]
Active block: [2, 3]
```

When `bitmask = False`, we get
```
field x[2, 2] = 0
field x[2, 3] = 2
field x[3, 2] = 0
field x[3, 3] = 0
field x[4, 6] = 0
field x[4, 7] = 0
field x[5, 6] = 3
field x[5, 7] = 0
Active block: [1, 1]
Active block: [2, 3]
```

Note that activating `x[2, 3]` also implicitly activates other pixels in `block[1, 1]`, e.g., `x[2, 2]`, `x[3, 2]`, and `x[3, 3]`.
Without a bitmask, these pixels in the same `block` share the same activity.

### Explicitly manipulating and querying sparsity

- Use `ti.is_active(snode, [i, j, ...])` to query if `snode[i, j, ...]` is active or not.
- `ti.activate/deactivate(snode, [i, j, ...])` to explicitly activate or deactivate a cell of `snode[i, j, ...]`. (TODO: example. Also note that `ti.activate` may have a different behavior...)
- Use `snode.deactivate_all()` to deactivate all cells of SNode `snode`.
- Use `ti.deactivate_all_snodes()` to deactivate all cells of all SNodes with sparsity.

:::note
For performance reasons, `ti.deactivate` ...
- does **not** recursively deactivate all the sub-cells of a cell.
- does **not** trigger a garbage of its parent cell, if all the children of the parent cell are deactivated.
- does **not** recursively deactivate its children cells.
:::

:::note
When deactivation happens, the Taichi runtime automatically recycles and zero-fills memory of the cells that are deactivated via `ti.deactivate`.
:::


### Finding parent cell indices

It is often helpful to compute the index of of a parent SNode cell given a child SNode cell.
While it is possible to directly use `[i // 2, j // 2]` to compute the `block` index given `pixel` index,
doing so couples computation code with the internal configuration of data structures (in this case, the size of `block` cells).

You can use `ti.rescale_index(TODO)`.

### Creating lists with the `dynamic` SNode

In particle simulations

- `ti.append(...)`
- `ti.length(...)`

How to clear a list?

## Further reading

Please read our [paper](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf),
watch the [introduction video](https://www.youtube.com/watch?v=wKw8LMF3Djo), or check out
the SIGGRAPH Asia 2019 [slides](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang-slides.pdf)
for more details on sparse computation.

[Taichi elements](https://github.com/taichi-dev/taichi_elements) implement a high-performance
MLS-MPM solver on Taichi's sparse grids.
