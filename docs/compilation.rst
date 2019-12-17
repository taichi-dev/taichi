The life of a Taichi kernel
===============================================

Sometimes it is helpful to understand the life cycle of a Taichi kernel.

In short, compilation will only happen on the first invocation of the kernel.

The compilation steps are: TODO: update

 - Function registration
 - AST transform
 - Template instantiation (if template kernels exist)
 - Taichi IR compilation and optimization and offloading
 - Kernel launching

Let's consider the following simple kernel:

.. code-block:: python

  @ti.kernel
  def add(tensor: ti.template(), delta: ti.i32):
    for i in tensor:
      tensor[i] += delta

Kernel Registration
---------------------------------------
When the function definition script is executed, the ``ti.kernel`` decorator registers the kernel.

Python Abstract Syntax Tree (AST) Transform
---------------------------------------
The first time the registered function is called, an AST transformer will transform the kernel body
into a Python script, which, when executed, emits a Taichi frontend AST.

Template Instantiation
---------------------------------------
.. code-block:: python

    @ti.kernel
    def

Template instantiation is lazy. Common use cases are...

Template signatures
---------------------

AST Lowering
-----------------------------------------

Taichi IR Optimization
-----------------------------------------
Access optimization etc


The Just-in-Time Compilation Engine
---------------------------------------
Finally, the optimized IR is fed into LLVM to generate executable CPU/GPU programs.

