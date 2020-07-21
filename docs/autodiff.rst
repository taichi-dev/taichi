.. _differentiable:

Differentiable programming
==========================

We suggest starting with the ``ti.Tape()``, and then migrate to more advanced differentiable programming using the ``kernel.grad()`` syntax if necessary.

Using ``ti.Tape()``
-------------------

For a physical simulation, sometimes it could be easy to compute the energy but
hard to compute the forces on each particles.

You can automatically differentiate (negative) potential energies to get forces.
Let's take `examples/ad_gravity.py <https://github.com/taichi-dev/taichi/blob/master/examples/ad_gravity.py>`_ as an example:

.. code-block:: python

    import taichi as ti
    ti.init()

    N = 8
    dt = 5e-5

    pos = ti.Vector.var(2, ti.f32, N, needs_grad=True)
    vel = ti.Vector.var(2, ti.f32, N)
    potential = ti.var(ti.f32, (), needs_grad=True)


    @ti.kernel
    def calc_potential():
        for i, j in ti.ndrange(N, N):
            disp = pos[i] - pos[j]
            potential[None] += 1 / disp.norm(1e-3)


    @ti.kernel
    def init():
        for i in pos:
            pos[i] = [ti.random(), ti.random()]


    @ti.kernel
    def advance():
        for i in pos:
            vel[i] += dt * pos.grad[i]
        for i in pos:
            pos[i] += dt * vel[i]


    def substep():
        with ti.Tape(potential):
            calc_potential()
        advance()


    init()
    gui = ti.GUI('Autodiff gravity')
    while gui.running and not gui.get_event(gui.ESCAPE):
        for i in range(16):
            substep()
        gui.circles(pos.to_numpy(), radius=3)
        gui.show()

Using ``kernel.grad()``
-----------------------

TODO: Documentation WIP.

.. _simplicity_rule:

Kernel Simplicity Rule
----------------------

Unlike tools such as TensorFlow where **immutable** output buffers are generated, the **imperative** programming paradigm adopted in Taichi allows programmers to freely modify global tensors.

To make automatic differentiation well-defined under this setting, we make the following assumption on Taichi programs for differentiable programming:

**Global Data Access Rules:**

  - If a global tensor element is written more than once, then starting from the second write, the write **must** come in the form of an atomic add (“accumulation", using ``ti.atomic_add`` or simply ``+=``).
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

Taichi programs that violate this rule will result in an error.

.. note::

  **static for-loops** (e.g. ``for i in ti.static(range(4))``) will get unrolled by the Python frontend preprocessor and therefore does not count as a level of loop.


DiffTaichi
----------

The `DiffTaichi repo <https://github.com/yuanming-hu/difftaichi>`_ contains 10 differentiable physical simulators built with Taichi differentiable programming.
A few examples with neural network controllers optimized using differentiable simulators and brute-force gradient descent:

.. image:: https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/ms3_final-cropped.gif

.. image:: https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/rb_final2.gif

.. image:: https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/diffmpm3d.gif

Check out `the DiffTaichi paper <https://arxiv.org/pdf/1910.00935.pdf>`_ and `video <https://www.youtube.com/watch?v=Z1xvAZve9aE>`_ to learn more about Taichi differentiable programming.
