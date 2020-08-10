Why new programming language
============================

Taichi is a high-performance programming language for computer graphics applications.

Introduction
------------

Different backends
******************

Before Taichi has been invented, to write a **general purposed GPU (GPGPU)**
program, we have to manipulate through tons of different APIs, e.g.:

``cuDeviceSynchronize()`` for CUDA, ``glDispatchCompute()`` for OpenGL,
``pthread_join()`` for multi-core CPUs...

Downside:

1. CUDA, for example, despite its powerfulness, can run only on supported NVIDIA
   GPUs. One could not share/distribute his/her work with/to a computer
   without NVIDIA GPUs. This means the loss of portability.

2. OpenGL, due to its historical debt, is too complicated for beginners when
   used as a GPGPU programming tool. People could easily get
   lost in API manuals instead of focusing on their original goals.
   This could cause the loss of productivity.

3. GPU programming langurages are often less flexible and less functional.
   It's easy to make use of high-level C++ features and external libraries
   on CPU. But it would be hard, at least in the OpenGL shader langurage,
   to do so.

4. It's relatively hard to visualize the results in GUI. You will have to
   make a bridge to your native GUI system before starting the debugging process.

5. It's very hard to optimize data structures for sparse computation accessing
   tensors / fields with plain indices. It would be great if we could decouple
   computations from data structures.

6. It is often the case that we need to compute the derivative of a kernel
   function, e.g. compute forces from gradients of potential energy. We will
   have to calculate the derivative by hand or external tools like Matlab.

7. A lot of repetitive work during study or research can be pretty distressing.
   This will not only waste a lot of programming time, but also kill off many
   beginner's interest. A highly integrated tool will significantly improve
   this situation.

8. Despite there are tools like Unity or Unreal providing a cross-backend
   computer graphics infrastructure, they are often aimed for game developers
   and artists, instead of **physics engine** developers, including industrial
   CFD applications.

All these reasons make it very hard for **computer graphics and computational
physics** beginners to write a GPGPU program.

Unified frontend
****************

But don't worry, here comes Taichi, a powerful tool aimed to make GPGPU
programming easier & accessible:

1. **Unify backend details**.
   Taichi is distributed as a Python package that ideally could run everywhere.
   For now it can supports CUDA, OpenGL 4.3+, Apple Metal, and 64-bit CPUs as
   backends. As long as you have one of them, you can run all Taichi programs!
   See :ref:`backends` for more details.

2. **General-purposed APIs**.
   Taichi provides a set of general-purposed API that provides many flexibility
   that could be easily applied in many fields including computer graphics,
   particle simulation, continuum physics, fluid dynamics, etc.
   So you won't bothered yourself with some ad-hoc usage of OpenGL fragment
   shaders :)

3. **Embedding in Python**.
   As we all know, Python is an interpreted programming langurage with many
   high-level features and cool packages. You can easily share data between
   Taichi and other Python packages like NumPy, PyTorch or matplotlib.

4. **Cross-platform GUI support**.
   Taichi have a built-in GUI support that can run on either Windows, OS X
   or X11 for Linux. You can visualize the computation result with very little
   effort. Not to say the external data visualizing packages like matplotlib.

5. **Decouple computation from data structures**.
   Taichi have a fundamental support on **sparse computation**, many
   optimization will be done by the Taichi compiler and complex sparsity
   becomes affordable. See :ref:`sparse` for more details.

6. **Differentiable programming**.
   Taichi have a built-in differator that is capable of compute
   the derivative of a kernel automatically. Once you have a kernel that
   computes potential energy, you can easily get the forces on each elements.
   See :ref:`differentiable` for more details.


Summary
-------

Features
********

To sum up, Taichi provides you with:

- Productivity
- Portability
- Performance
- Metaprogramming
- Objective data-oriented programming
- Spatially sparse computation
- Differentiable programming

Design decisions
****************

- Decouple computation from data structures
- Domain-specific compiler optimizations
- Customizable megakernels
- Two-scale automatic differentiation
- Embedding in Python

Example
*******

Here we showcase an example on how to render a classical UV image via Taichi:

.. code-block:: python

    import taichi as ti  # make sure you've 'pip3 install taichi' already

    # declare a 512x512x3 field whose elements are 32-bit floating-point numbers
    rgb_image = ti.field(dtype=float, shape=(512, 512, 3))


    @ti.kernel  # functions decorated by @ti.kernel will be compiled by Taichi
    def render():
        # iterate through 512x512 pixels in parallel
        for i, j in ti.ndrange(512, 512):
            r = i / 512
            g = j / 512
            rgb_image[i, j, 0] = r  # red channel, from 0.0 to 1.0
            rgb_image[i, j, 1] = g  # green channel, from 0.0 to 1.0


    gui = ti.GUI('UV', (512, 512))  # create a 512x512 window
    while gui.running:
        render()
        gui.set_image(rgb_image)  # display the field as an image
        gui.show()


See :ref:`install` for more details about how to install Taichi via ``pip``.

See :ref:`hello` for more details about Taichi langurage and syntax.
