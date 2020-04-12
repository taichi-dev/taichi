Hello, world!
===============================================

We introduce the Taichi programming language through a very basic `fractal` example.

If you haven't done so, please install Taichi via ``pip``.
Depending on your hardware and OS, please execute one of the following commands:

.. code-block:: bash

  # python 3.6+ required

  python3 -m pip install taichi-nightly


Now you are ready to run the Taichi code below (``python3 fractal.py``) to compute a
`Julia set <https://en.wikipedia.org/wiki/Julia_set>`_:

.. image:: https://github.com/yuanming-hu/public_files/raw/master/graphics/taichi/fractal.gif

.. code-block:: python

  # fractal.py

  import taichi as ti

  ti.init(arch=ti.cuda) # Run on GPU by default

  n = 320
  pixels = ti.var(dt=ti.f32, shape=(n * 2, n))

  @ti.func
  def complex_sqr(z):
    return ti.Vector([z[0] * z[0] - z[1] * z[1], z[1] * z[0] * 2])

  @ti.kernel
  def paint(t: ti.f32):
    for i, j in pixels: # Parallized over all pixels
      c = ti.Vector([-0.8, ti.sin(t) * 0.2])
      z = ti.Vector([float(i) / n - 1, float(j) / n - 0.5]) * 2
      iterations = 0
      while z.norm() < 20 and iterations < 50:
        z = complex_sqr(z) + c
        iterations += 1
      pixels[i, j] = 1 - iterations * 0.02

  gui = ti.GUI("Fractal", (n * 2, n))

  for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()



Let's dive into components of this simple Taichi program.

import taichi as ti
-------------------
Taichi is an embedded domain-specific language (DSL) in Python.
It pretends to be a plain Python package, although heavy engineering has been done to make this happen.

This design decision virtually makes every Python programmer capable of writing Taichi programs, after minimal learning efforts.
You can also reuse the package management system, Python IDEs, and existing Python packages.

Portability
-----------------

Taichi supports CPUs, NVIDIA GPUs, Mac GPUs with Metal, or any GPUs with OpenGL API.

.. code-block:: python

  # Run on NVIDIA GPU, require CUDA installed
  ti.init(arch=ti.cuda)
  # Run on GPU, with OpenGL backend
  ti.init(arch=ti.opengl)
  # Run on GPU, with Metal backend, if you're on OS X
  ti.init(arch=ti.metal)
  # Run on CPU (default)
  ti.init(arch=ti.x64)

.. note::
    Supported archs on different platforms:

    +----------+------+------+--------+-------+
    | platform | CPU  | CUDA | OpenGL | Metal |
    +==========+======+======+========+=======+
    | Windows  | OK   | WIP  | WIP    | MISS  |
    +----------+------+------+--------+-------+
    | Linux    | OK   | OK   | OK     | MISS  |
    +----------+------+------+--------+-------+
    | Mac OS X | OK   | MISS | MISS   | OK    |
    +----------+------+------+--------+-------+

    (OK=supported, WIP=work in progress, MISS=not supported)

If the machine does not have CUDA support, Taichi will fall back to CPUs instead.

.. note::

  When running the CUDA backend on Windows and ARM devices (e.g. NVIDIA Jetson),
  Taichi will by default allocate 1 GB memory for tensor storage. You can override this by initializing with
  ``ti.init(arch=ti.cuda, device_memory_GB=3.4)`` to allocate ``3.4`` GB GPU memory, or
  ``ti.init(arch=ti.cuda, device_memory_fraction=0.3)`` to allocate ``30%`` of total available GPU memory.

  On other platforms Taichi will make use of its on-demand memory allocator to adaptively allocate memory.

(Sparse) Tensors
----------------

Taichi is a data-oriented programming language, where dense or spatially-sparse tensors are first-class citizens.
See :ref:`sparse` for more details on sparse tensors.

``pixels = ti.var(dt=ti.f32, shape=(n * 2, n))`` allocates a 2D dense tensor named ``pixel`` of
size ``(640, 320)`` and type ``ti.f32`` (i.e. ``float`` in C).

Functions and kernels
---------------------

Computation happens within Taichi **kernels**. Kernel arguments must be type-hinted.
The language used in Taichi kernels and functions looks exactly like Python, yet the Taichi frontend compiler converts it
into a language that is **compiled, statically-typed, lexically-scoped, parallel, and differentiable**.

You can also define Taichi **functions** with ``ti.func``, which can be called and reused by kernels and other functions.

.. note::

  **Taichi-scope v.s. Python-scope**: everything decorated with ``ti.kernel`` and ``ti.func`` is in Taichi-scope, which will be compiled by the Taichi compiler.
  Code outside the Taichi-scopes is simply native Python code.

.. warning::

  Taichi kernels must be called in the Python-scope. I.e., **nested Taichi kernels are not supported**.
  Nested functions are allowed. **Recursive functions are not supported for now**.

  Taichi functions can only be called in Taichi-scope.

For those who came from the world of CUDA, ``ti.func`` corresponds to ``__device__``, and ``ti.kernel`` corresponds to ``__global__``.


Parallel for-loops
-----------------------
For loops at the outermost scope in a Taichi kernel is automatically parallelized.
For loops can have two forms, i.e. `range-for loops` and `struct-for loops`.

**Range-for loops** are no different from that in native Python, except that it will be parallelized
when used as the outermost scope. Range-for loops can be nested.

.. code-block:: python

  @ti.kernel
  def fill():
    for i in range(10): # parallelized
      x[i] += i

      s = 0
      for j in range(5): # serialized in each parallel thread
        s += j

      y[i] = s

  @ti.kernel
  def fill_3d():
    # Parallelized for all 3 <= i < 8, 1 <= j < 6, 0 <= k < 9
    for i, j, k in ti.ndrange((3, 8), (1, 6), 9):
      x[i, j, k] = i + j + k

**Struct-for loops** have a cleaner syntax, and are particularly useful when iterating over tensor elements.
In the fractal code above, ``for i, j in pixels`` loops over all the pixel coordinates, i.e. ``(0, 0), (0, 1), (0, 2), ... , (0, 319), (1, 0), ..., (639, 319)``.

.. note::

    Struct-for is the key to :ref:`sparse` in Taichi, as it will only loop over active elements in a sparse tensor. In dense tensors, all elements are active.

.. note::
    It is the loop **at the outermost scope** that gets parallelized, not the outermost loop.

    .. code-block:: python

      # Good kernel
      @ti.func
      def foo():
        for i in x:
          ...

      # Bad kernel
      @ti.func
      def bar(k: ti.i32):
        # The outermost scope is a `if` statement, not the struct-for loop!
        if k > 42:
          for i in x:
            ...

.. note::
    ``break`` is not supported in **outermost (parallelized)** loops:

    .. code-block:: python

      @ti.kernel
      def foo():
        for i in x:
            ...
            break # ERROR! You cannot break a parallelized loop!

      @ti.kernel
      def foo():
        for i in x:
            for j in y:
                ...
                break # OK


Interacting with Python
------------------------

Everything outside Taichi-scope (``ti.func`` and ``ti.kernel``) is simply Python. You can use your favorite Python packages (e.g. ``numpy``, ``pytorch``, ``matplotlib``) with Taichi.

In Python-scope, you can access Taichi tensors using plain indexing syntax, and helper functions such as ``from_numpy`` and ``to_torch``:

.. code-block:: python

  image[42, 11] = 0.7
  print(image[1, 63])

  import numpy as np
  pixels.from_numpy(np.random.rand(n * 2, n))

  import matplotlib.pyplot as plt
  plt.imshow(pixels.to_numpy())
  plt.show()
