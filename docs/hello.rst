Hello, world!
===============================================

We introduce the Taichi programming language through this very basic example.

If you haven't done so, please install Taichi via ``pip``.
Depending on your hardware and OS, please execute one of the following commands.

.. code-block:: bash

  # CPU only. No GPU/CUDA needed. (Linux, OS X and Windows)
  python3 -m pip install taichi-nightly

  # With GPU (CUDA 10.0) support (Linux only)
  python3 -m pip install taichi-nightly-cuda-10-0

  # With GPU (CUDA 10.1) support (Linux only)
  python3 -m pip install taichi-nightly-cuda-10-1

Now you are ready to use Taichi.

.. code-block:: python

  import taichi as ti

  n = 320
  pixels = ti.var(dt=ti.f32, shape=(n * 2, n))
  ti.cfg.arch = ti.cuda # Run on GPU by default

  @ti.func
  def complex_sqr(z):
    return ti.Vector([z[0] * z[0] - z[1] * z[1], z[1] * z[0] * 2]) # z^2

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

  gui = ti.core.GUI("Fractal", ti.veci(n * 2, n))

  for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.update()

Running this code gives you

.. image:: https://github.com/yuanming-hu/public_files/raw/master/graphics/taichi/fractal.gif

Now let's dive into components of this simple Taichi program.

import taichi as ti
-------------------
Taichi is embedded in Python.
This virtually makes every Python programmer capable of writing Taichi programs, with minimal learning efforts required.

Functions and kernels
---------------------
Functions can be called in kernels,

The outermost for-loop in a Taichi kernel is automatically parallelized.

Range-for v.s. struct-for
----------------------------------
.. warning::

    Struct-for's must be at the outer-most level of kernels. Struct-for's cannot be nested.


Interaction with Python
------------------------
Everything outside the scope of ``ti.func`` and ``ti.kernel`` is natively Python. Nothing new.

You can use your favourite Python packages (e.g. ``numpy``, ``pytorch``, ``matplotlib``) with Taichi.

