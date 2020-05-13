.. _differentiable:

Differentiable programming
==========================

This page is work in progress. Please check out `the DiffTaichi paper <https://arxiv.org/pdf/1910.00935.pdf>`_ and `video <https://www.youtube.com/watch?v=Z1xvAZve9aE>`_ to learn more about Taichi differentiable programming.

The `DiffTaichi repo <https://github.com/yuanming-hu/difftaichi>`_ contains 10 differentiable physical simulators built with Taichi differentiable programming.

.. note::
    Unlike tools such as TensorFlow where **immutable** output buffers are generated,
    the **imperative** programming paradigm adopted in Taichi allows programmers to freely modify global tensors.
    To make automatic differentiation well-defined under this setting,
    we make the following assumption on Taichi programs for differentiable programming:

    **Global Data Access Rules:**

      - If a global tensor element is written more than once, then starting from the second write, the
        write **must** come in the form of an atomic add (“accumulation", using ``ti.atomic_add`` or simply ``+=``).
      - No read accesses happen to a global tensor element, until its accumulation is done.

    **Kernel Simplicity Rule:** Kernel body consists of multiple `simply nested` for-loops.
    I.e., each for-loop can either contain exactly one (nested) for-loop (and no other statements), or a group of statements without loops.

    Example:

    .. code-block:: python

        @ti.kernel
        def differentiable_task():
          for i in x:
            x[i] = y[i]

          for i in range(10):
            for j in range(20):
              for k in range(300):
                ... do whatever you want, as long as there are no loops

          # Not allowed. The outer for loop contains two for loops
          for i in range(10):
            for j in range(20):
              ...
            for j in range(20):
              ...

    Taichi programs that violate this rule has an undefined gradient behavior.

.. note::

  **static for-loops** (e.g. ``for i in ti.static(range(4))``) will get unrolled by the Python frontend preprocessor and does not count as a level of loop.


A few examples with neural network controllers optimized using differentiable simulators and brute-force gradient descent:

.. image:: https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/ms3_final-cropped.gif

.. image:: https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/rb_final2.gif

.. image:: https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/diffmpm3d.gif

Documentation WIP.
