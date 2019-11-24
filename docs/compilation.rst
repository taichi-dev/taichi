The Life of a Taichi Kernel
===============================================

Sometimes it is helpful to understand the life cycle of a Taichi kernel.

In short, compilation will only happen on the first invocation of the kernel.

The steps are:
 - Template instantiation (if template kernels exist)
 - AST transform
 - Taichi IR compilation and optimization and offloading
 - Kernel launching

Python Abstract Syntax Tree (AST) Transform
---------------------------------------



Template Instantiation
---------------------------------------
.. code-block:: python

    @ti.kernel
    def


Template instantiation is lazy. Common use cases are
 -

The Just-in-Time Compilation Engine
---------------------------------------
Finally, the optimized IR is fed into LLVM to generate executable CPU/GPU programs.

