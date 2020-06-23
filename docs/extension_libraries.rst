Extension libraries
===================

The Taichi programming language offers a minimal and generic built-in standard library. Extra domain-specific functionalities are provided via **extension libraries**:


Taichi GLSL
-----------

`Taichi GLSL <https://github.com/taichi-dev/taichi_glsl>`_ is an extension
library of Taichi, aiming at providing useful helper functions including:

1. Handy scalar functions like ``clamp``, ``smoothstep``, ``mix``, ``round``.
2. GLSL-alike vector functions like ``normalize``, ``distance``, ``reflect``.
3. Well-behaved random generators including ``randUnit2D``, ``randNDRange``.
4. Handy vector and matrix initializer: ``vec`` and ``mat``.
5. Handy vector component shuffle accessor like ``v.xy``.

Click here for `Taichi GLSL Documentation <https://taichi-glsl.readthedocs.io>`_.

.. code-block:: bash

    python3 -m pip install taichi_glsl

.. code-block:: python

    from taichi_glsl import *
    import time, math

    image = vec_array(3, float, 512, 512)

    @ti.kernel
    def paint(t: ti.f32):
        for i, j in image:
            coor = view(image, i, j)
            image[i, j] = smoothstep(distance(coor, vec(0.5, 0.5)), t,
                                     t - 0.06) * vec(coor.x, coor.y, 0.0)

    with ti.GUI('Step UV') as gui:
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            paint(0.4 + 0.4 * math.cos(time.time()))
            gui.set_image(image)
            gui.show()


Taichi THREE
------------

`Taichi THREE <https://github.com/taichi-dev/taichi_three>`_ is an extension
library of Taichi to render 3D scenes into nice-looking 2D images in real-time
(work in progress).

.. image:: https://raw.githubusercontent.com/taichi-dev/taichi_three/16d98cb1c1f2ab7a37c9e42260878c047209fafc/assets/monkey.png

Click here for `Taichi THREE Tutorial <https://github.com/taichi-dev/taichi_three#how-to-play>`_.

.. code-block:: bash

    python3 -m pip install taichi_three

.. code-block:: python

    import taichi as ti
    import taichi_three as t3
    import numpy as np

    ti.init()

    scene = t3.Scene()
    model = t3.Model()
    scene.add_model(model)

    vertices = t3.Vertex.var(3)
    faces = t3.Face.var(2)
    vertices.pos.from_numpy(np.array(
        [[+0.0, +0.5, 0.0], [-0.5, -0.5, 0.0], [+0.5, -0.5, 0.0]]))
    faces.idx.from_numpy(np.array([[0, 1, 2], [0, 2, 1]])) # both cull faces
    model.set_vertices(vertices)
    model.add_geometry(faces)

    scene.set_light_dir([0.4, -1.5, -1.8])
    gui = ti.GUI('Triangle', scene.res)
    while gui.running:
        gui.running = not gui.get_event(ti.GUI.ESCAPE)
        scene.camera.from_mouse(gui)
        #model.L2W.from_mouse(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.show()
