---
sidebar_position: 9
---

# Extension libraries

The Taichi programming language offers a minimal and generic built-in
standard library. Extra domain-specific functionalities are provided via
**extension libraries**:

## Taichi GLSL

[Taichi GLSL](https://github.com/taichi-dev/taichi_glsl) is an extension
library of Taichi, aiming at providing useful helper functions
including:

1.  Handy scalar functions like `clamp`, `smoothstep`, `mix`, `round`.
2.  GLSL-alike vector functions like `normalize`, `distance`, `reflect`.
3.  Well-behaved random generators including `randUnit2D`,
    `randNDRange`.
4.  Handy vector and matrix initializer: `vec` and `mat`.
5.  Handy vector component shuffle accessor like `v.xy`.

Click here for [Taichi GLSL
Documentation](https://taichi-glsl.readthedocs.io).

```bash
python3 -m pip install taichi_glsl
```

## Taichi THREE

[Taichi THREE](https://github.com/taichi-dev/taichi_three) is an
extension library of Taichi to render 3D scenes into nice-looking 2D
images in real-time (work in progress).

<center>

![image](https://raw.githubusercontent.com/taichi-dev/taichi_three/16d98cb1c1f2ab7a37c9e42260878c047209fafc/assets/monkey.png)

</center>

Click here for [Taichi THREE
Tutorial](https://github.com/taichi-dev/taichi_three#how-to-play).

```bash
python3 -m pip install taichi_three
```
