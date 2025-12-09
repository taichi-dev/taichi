.. meta::
  :description: What is Taichi Lang?
  :keywords: Taichi Lang, python, programming, JIT, AMD, ROCm, overview, introduction

.. _what-is-taichi:

********************************************************************
What is Taichi Lang?
********************************************************************

Taichi Lang is an open-source, imperative, and parallel programming language embedded
in Python, designed for high-performance numerical computation and real-time physical
simulation. It uses just-in-time (JIT) compilation frameworks such as LLVM to accelerate
compute-intensive Python code by compiling it into optimized GPU or CPU instructions.

Features and use cases
====================================================================

Taichi Lang allows developers to write concise, high-level algorithms while leaving
performance optimization to Taichi’s compiler. Taichi Lang is widely used in domains such
as fluid dynamics, particle-based simulations, robotics, computer vision, augmented
reality, artificial intelligence, and visual effects for film and gaming. For example,
simulating a cloth falling onto a sphere, a system with tens of thousands of mass
points and springs, can be implemented in Taichi Lang with only a few dozen lines of
Python code due to its data-oriented design and automatic parallelization.

Taichi Lang also supports advanced techniques such as hierarchical sparse voxel grids for
large-scale simulations, enabling efficient handling of spatially sparse data structures
in 3D visual computing. On ROCm, Taichi Lang is officially supported on AMD Instinct
GPUs, making it a powerful tool for developers who need both flexibility and performance.


Why Taichi Lang?
====================================================================

- **Built around Python**: Taichi Lang shares almost the same syntax with Python, 
  which lets you write algorithms with minimal syntax differences. It is also well
  integrated into the Python ecosystem, including NumPy and PyTorch.

- **Flexibility**: Taichi Lang provides a set of generic data containers known 
  as ``SNode``, an effective mechanism for composing hierarchical, 
  multi-dimensional fields. This can cover many use patterns in numerical simulation
  (for example, `spatially sparse computing <https://docs.taichi-lang.org/docs/sparse>`__).

- **Performance**: With the ``ti.kernel`` decorator, Taichi Lang's JIT compiler 
  automatically compiles your Python functions into efficient GPU or CPU machine
  code for parallel execution.

- **Portability**: Write your code once and run it everywhere. Taichi 
  Lang supports most mainstream GPU APIs, such as NVIDIA CUDA and Vulkan. You can
  write your code with Taichi Lang on ROCm and use it extensibly.

- **Additional feature support**: Cross-platform support that includes a Vulkan-based 3D visualizer,
  `differentiable programming <https://docs.taichi-lang.org/docs/differentiable_programming>`__,
  `quantized computation <https://github.com/taichi-dev/quantaichi>`__ (experimental), and many more.
