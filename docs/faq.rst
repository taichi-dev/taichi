Frequently asked questions
==========================

**Can a user iterate over irregular topology instead of grids, such as tetrahedra meshes, line segment vertices?**
These structures have to be represented using 1D arrays in Taichi. You can still iterate over it using `for i in x` or `for i in range(n)`.
However, at compile time, there's little the Taichi compiler can do for you to optimize it. You can still tweak the data layout to get different run time cache behaviors and performance numbers.

**Can potential energies be differentiated automatically to get forces?**
Yes. Taichi supports automatic differentiation.
We do have an `example <https://github.com/yuanming-hu/taichi/blob/master/examples/mpm_lagrangian_forces.py>`_ for this.

**Does the compiler backend support the same quality of optimizations for the GPU and CPU? For instance, if I switch to using the CUDA backend, do I lose the cool hash-table optimizations?**
Mostly. The CPU/GPU compilation workflow are basically the same, except for vectorization on SIMD CPUs.
You still have the hash table optimization on GPUs.
