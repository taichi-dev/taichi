External libraries
==================

The Taichi Programming Language provides a set of built-in standard library.
The Taichi built-in library is aimed to be simple for maintainability, and
therefore generic and less domain specific.

However users may find that the built-in library along may not enough,
so here we provide some external libraries that hopefully could extend your
productivity in Taichi.


Taichi GLSL
-----------

`Taichi GLSL <https://github.com/taichi-dev/taichi_glsl>`_ is an external
library of Taichi, aimed to provide a set of useful helper functions including
but not limited to:

1. Handy scalar functions like ``clamp``, ``smoothstep``, ``mix``, ``round``.
2. GLSL-alike vector functions like ``normalize``, ``distance``, ``reflect``.
3. Well-behaved random generators including ``randUnit2D``, ``randNDRange``.
4. Possible Taichi BUG hotfixes that are not yet released in it's cycle.
5. Handy vector and matrix initializer: ``vec`` and ``mat``.

Click here for `Taichi GLSL Documentation <https://taichi-glsl.readthedocs.io>`_.


Taichi Elements
---------------

`Taichi Elements <https://github.com/taichi-dev/taichi_elements>`_ is a
High-Performance Multi-Material Continuum Physics Engine (work in progress).

The solver is being developed using Taichi, therefore it is cross-platform and
supports multithreaded CPUs and massively parallel GPUs.

The short-term plan is:

- To build a reusable multimaterial (water/elastic/snow/sand/mud) simulator
- To integrate the simulator into Blender

Click here for `Taichi Elements Documentation <https://taichi-elements.readthedocs.io>`_.
