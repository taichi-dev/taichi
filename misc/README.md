# FAQ

 - Q: Do you have a FAQ about Taichi available? 

   A: Here it is :-)


  - Q: Can a user iterate over irregular topology instead of grids, such as tetrahedra meshes, line segment vertices?

    A: These structures have to be represented using 1D arrays in Taichi. You can still iterate over it using `for i in x` or `for i in range(n)`.
However, there's little the Taichi compiler can do for you to optimize it.

  - Q: Can energies be differentiated natively?

    A: Yes. We do have an [example](https://github.com/yuanming-hu/taichi/blob/master/examples/mpm_lagrangian_forces.py) for this. 

  - Q: If so, I assume this must be some sort of autodiff process?

    A: Yes, we have an autodiff engine in Taichi.

  - Q: Does the compiler backend support the same quality of optimizations for the GPU and CPU?
 
    A: Mostly. The CPU/GPU compilation workflow are basically the same, except for vectorization on SIMD CPUs. 

  - Q: For instance, if I switch to using the CUDA backend, do I lose the cool hash-table optimizations? 

    A: You still have the hask table optimization on GPUs.
