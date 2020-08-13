.. _differentiable:

Differentiable programming
==========================

We suggest starting with the ``ti.Tape()``, and then migrate to more advanced differentiable programming using the ``kernel.grad()`` syntax if necessary.


Introduction
------------

For example, you have the following kernel:

.. code-block:: python

    x = ti.var(ti.f32, ())
    y = ti.var(ti.f32, ())

    @ti.kernel
    def compute_y():
        y[None] = ti.sin(x[None])


Now if you want to get the derivative of y corresponding to x, i.e., dy/dx.
You may want to implement the derivative kernel by yourself:

.. code-block:: python

    x = ti.var(ti.f32, ())
    y = ti.var(ti.f32, ())
    dy_dx = ti.var(ti.f32, ())

    @ti.kernel
    def compute_dy_dx():
        dy_dx[None] = ti.cos(x[None])


But wait, what if I changed the original ``compute_y``? We will have to
recalculate the derivative by hand and rewrite ``compute_dy_dx`` again, which
is very error-prone and not convenient at all.

If this situation occurs, don't worry! Taichi provides a handy autodiff
system that can help you obtain the derivative of a kernel without any pain!


Using ``ti.Tape()``
-------------------

Let's still take the ``compute_y`` in above example for explaination.
What's the most convienent way to obtain a kernel that computes x to dy/dx?

1. Use the ``needs_grad=True`` option when declaring fields involved in the
   derivative chain.
2. Use ``with ti.Tape(y):`` to embrace the invocation into kernel(s) you want
   to compute derivative.
3. Now ``x.grad[None]`` is the dy/dx value at current x.

.. code-block:: python

    x = ti.var(ti.f32, (), needs_grad=True)
    y = ti.var(ti.f32, (), needs_grad=True)

    @ti.kernel
    def compute_y():
        y[None] = ti.sin(x[None])

    with ti.Tape(y):
        compute_y()

    print('dy/dx =', x.grad[None])
    print('at x =', x[None])


It's equivalant to:

.. code-block:: python

    x = ti.var(ti.f32, ())
    y = ti.var(ti.f32, ())
    dy_dx = ti.var(ti.f32, ())

    @ti.kernel
    def compute_dy_dx():
        dy_dx[None] = ti.cos(x[None])

    compute_dy_dx()

    print('dy/dx =', dy_dx[None])
    print('at x =', x[None])


Usage example
+++++++++++++

For a physical simulation, sometimes it could be easy to compute the energy but
hard to compute the force on each particles.

But recall that we can differentiate (negative) potential energy to get forces.
a.k.a.: ``F_i = -dU / dx_i``.
So once you've write a kernel that is able to compute the potential energy,
you may use Taichi's autodiff system to obtain the derivative of it and
then the force on each particles.

Take `examples/ad_gravity.py <https://github.com/taichi-dev/taichi/blob/master/examples/ad_gravity.py>`_ as an example:

.. code-block:: python

    import taichi as ti
    ti.init()

    N = 8
    dt = 1e-5

    x = ti.Vector.var(2, ti.f32, N, needs_grad=True)  # position of particles
    v = ti.Vector.var(2, ti.f32, N)                   # velocity of particles
    U = ti.var(ti.f32, (), needs_grad=True)           # potential energy


    @ti.kernel
    def compute_U():
        for i, j in ti.ndrange(N, N):
            r = x[i] - x[j]
            # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
            # This is to prevent 1/0 error which can cause wrong derivative
            U[None] += -1 / r.norm(1e-3)  # U += -1 / |r|


    @ti.kernel
    def advance():
        for i in x:
            v[i] += dt * -x.grad[i]  # dv/dt = -dU/dx
        for i in x:
            x[i] += dt * v[i]        # dx/dt = v


    def substep():
        with ti.Tape(U):
            # every kernel invocation within this indent scope
            # will also be accounted into the partial derivate of U
            # with corresponding input variables like x.
            compute_U()   # will also computes dU/dx and save in x.grad
        advance()


    @ti.kernel
    def init():
        for i in x:
            x[i] = [ti.random(), ti.random()]


    init()
    gui = ti.GUI('Autodiff gravity')
    while gui.running:
        for i in range(50):
            substep()
        print('U = ', U[None])
        gui.circles(x.to_numpy(), radius=3)
        gui.show()


.. note::

   The argument ``U`` to ``ti.Tape(U)`` must be a 0D field.

   For using autodiff with multiple output variables, please see the
   ``kernel.grad()`` usage below.

.. note::

   ``ti.Tape(U)`` will automatically set `U[None]`` to 0 on start up.


See `examples/mpm_lagrangian_forces.py <https://github.com/taichi-dev/taichi/blob/master/examples/mpm_lagrangian_forces.py>`_ and `examples/fem99.py <https://github.com/taichi-dev/taichi/blob/master/examples/fem99.py>`_ for examples on using autodiff for MPM and FEM.


Using ``kernel.grad()``
-----------------------

TODO: Documentation WIP.


.. _simplicity_rule:

Kernel Simplicity Rule
----------------------

Unlike tools such as TensorFlow where **immutable** output buffers are generated, the **imperative** programming paradigm adopted in Taichi allows programmers to freely modify global fields.

To make automatic differentiation well-defined under this setting, we make the following assumption on Taichi programs for differentiable programming:

**Global Data Access Rules:**

  - If a global field element is written more than once, then starting from the second write, the write **must** come in the form of an atomic add (“accumulation", using ``ti.atomic_add`` or simply ``+=``).
  - No read accesses happen to a global field element, until its accumulation is done.

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
