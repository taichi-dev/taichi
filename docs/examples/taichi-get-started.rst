.. meta::
  :description: Get started using Taichi Lang
  :keywords: Taichi Lang, JIT, parallel, programming, python, ROCm, example, sample, tutorial

.. _taichi-get-started:

********************************************************************
Get started with Taichi Lang
********************************************************************

When writing compute-intensive tasks, you can make use of the two 
decorators ``@ti.func`` and ``@ti.kernel``. Functions decorated 
with ``@ti.kernel`` are kernels that serve as the entry points where
Taichi’s runtime takes over the tasks, and they must be directly
invoked by Python code. Functions decorated with ``@ti.func`` are
building blocks of kernels and can only be invoked by another
Taichi Lang function or a kernel. These decorators instruct Taichi
Lang to take over the computation tasks and compile the decorated
functions to machine code using just-in-time (JIT) compiler. As a
result, calls to these functions are executed on multi-core CPUs
or GPUs.

Below you can see how simple it is to use the Taichi Lang ``@ti.func``
and ``@ti.kernel`` decorators to accelerate Python code. First, you 
can see the Python code without using Taichi Lang. To enact Taichi Lang 
in this example, the function ``inv_square`` serves as a building 
block function for the kernel ``partial_sum``.

The example Python code without Taichi Lang:

.. code-block:: bash

    def inv_square(x):  # A function
        return 1.0 / (x * x)

    def partial_sum(n: int) -> float:  # A kernel
        total = 0.0
        for i in range(1, n + 1):
            total += inv_square(i)
        return total

    partial_sum(1000)

1. To write this code in Taichi Lang, import and initialize Taichi Lang for code acceleration:

.. code-block:: bash
   
   import taichi as ti
   ti.init(arch=ti.gpu)

2. Then, you can decorate the building block function and kernel with the ``@ti.func`` and ``@ti.kernel`` decorators, respectively:

.. code-block:: bash

   @ti.func
   def inv_square(x):  # A Taichi function
       return 1.0 / (x * x)

   @ti.kernel
   def partial_sum(n: int) -> float:  # A kernel
       total = 0.0
       for i in range(1, n + 1):
           total += inv_square(i)
       return total