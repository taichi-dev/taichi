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
4. Possible Taichi BUG hotfixes that are not yet released in it's cycle.
5. Handy vector and matrix initializer: ``vec`` and ``mat``.

Click here for `Taichi GLSL Documentation <https://taichi-glsl.readthedocs.io>`_.
