---
sidebar_position: 3
---

# Sparse computation

## Motivation

High-resolution 2D/3D grids are often needed in large-scale spatial computation, especially physical simulation.
However, they tend to take a huge amount of memory space and computation.

While a programmer may allocate large dense grids to store spatial data (especially physical quantities such as a density field),
oftentimes he or she only cares about a small fraction of this dense grid, the rest being ambient space (air).

In short, the regions of interest may only occupy a small fraction of the bounding volume.
If we can leverage such "spatial sparsity" and focus computation on the regions we care about,
we will significantly save storage and computing power.

:::note
The key to leverage spatial sparsity is to replace dense grids with sparse grids.
:::

Here is a simple example. The 2D multi-physics (material point method) simulation below has 256x256 grid cells.
Since the simulated objects do not fully occupy the whole domain, we would like to *adaptively* allocate the underlying simulation grid.
We subdivide the whole simulation domain into 16x16 *blocks*,
and each block has 16x16 *grid cells*.
Memory allocation can then happen at *block* granularity,
and we only consume memory space of blocks that are actually in the simulation.

![image](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi_elements/sparse_mpm_active_blocks.gif)

(Note the changing distribution of blocks throughout the simulation.)

:::note
**Backend compatibility**: The LLVM backends (CPU/CUDA) and the Metal backend offer the full functionality of sparse computation.
Other backends offer no or limited support of sparse computation.
:::

:::note
Sparse matrices are usually **not** implemented in Taichi via (spatially-) sparse data structures.
:::

## Sparse data structures in Taichi

### Data structure hierarchy

To concentrate computation on sparse regions of interest, multilevel sparse voxel data structures are studied extensively. 

## Activation on write

When writing to an inactive cell on a sparse data structre, Taichi automatically populates the data structure.

![image](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/sparse_grids.gif)

## Sparse struct-fors

Efficiently looping over sparse grid cells that are irregular can be a challenge. Fortunately, in Taichi, *struct-for's*
natively support sparse data structures and only loops over voxels that are currently active.

## Explicitly manipulating and querying sparsity

- `ti.is_active(...)`
- `ti.activate(...)`
- `SNode.deactivate_all()`
- `ti.deactivate_all_snodes()`

### Automatic garbage collection

## Further reading

Please read our [paper](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf),
watch the [introduction video](https://www.youtube.com/watch?v=wKw8LMF3Djo), or check out
the SIGGRAPH Asia 2019 [slides](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang-slides.pdf)
for more details on sparse computation.

[Taichi elements](https://github.com/taichi-dev/taichi_elements) implement a high-performance
MLS-MPM solver on Taichi's sparse grids.
