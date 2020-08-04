Frequently asked questions
==========================


**Q:** Installing Taichi with ``pip``, complains ``package not found``.

**A:** Is your Python version >= 3.6, and 64-bit? See :ref:`troubleshooting`.

**Q:** Do we have something like ``ti.pi``?

**A:** No, but you may use ``math.pi`` or ``numpy.pi`` instead. Taichi is able to bake in these constants during JIT, so your kernels incur no runtime cost.

**Q:** How do I **force** an outermost loop to be serial, i.e. **not parallelized**?

**A:** Try this trick:

.. code-block:: python

    for _ in range(1):  # I'm the outer-most loop!
        for i in range(100):  # This loop will not be parallelized
            ...

**Q:** What's the most convenient way to load images / textures into Taichi fields?

**A:** Simply use ``field.from_numpy(ti.imread('filename.png'))``.

**Q:** Can Taichi co-operate with **other Python packages** like ``matplotlib``?

**A:** Yes, as long as that *package* provides an interface with ``numpy``, see :ref:`other_python_packages`.

**Q:** Shall we add some handy functions like ``ti.smoothstep`` or ``ti.vec3``?

**A:** No, but we provide them in an extension library `Taichi GLSL <https://taichi-glsl.readthedocs.io>`_ , install it using:

.. code-block:: bash

    python -m pip install taichi_glsl

**Q:** How can I **render 3D results** without writing a ray tracer myself?

**A:** You may export it with :ref:`export_ply_files` so that you could view it in Houdini or Blender.
       Or make use the extension library `Taichi THREE <https://github.com/taichi-dev/taichi_glsl>`_ to render images and update to GUI in real-time.

**Q:** How do I declare a field with **dynamic length**?

**A:** What you want may be the ``dynamic`` SNode, a kind of sparse field, see :ref:`dynamic`.
       Or simply allocate a dense field large enough, and another 0-D field ``field_len[None]`` for length record.
       But in fact, the ``dynamic`` SNode could be slower than the latter solution, due to the cost of maintaining the sparsity information.

**Q:** Can a user iterate over irregular topologies (e.g., graphs or tetrahedral meshes) instead of regular grids?

**A:** These structures have to be represented using 1D arrays in Taichi. You can still iterate over them using ``for i in x`` or ``for i in range(n)``.
       However, at compile time, there's little the Taichi compiler can do for you to optimize it. You can still tweak the data layout to get different runtime cache behaviors and performance numbers.
