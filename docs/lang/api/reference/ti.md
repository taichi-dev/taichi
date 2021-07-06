---
title: ti
docs:
  desc: |
    This page documents functions and properties under `ti` (assume you have done `import taichi as ti`) namespace.
  functions:
    - name: block_dim
      desc: |
        A decorator to tweak the property of a for-loop, specify the threads per block of the next parallel for-loop.

        :::note
        The argument `n` must be a power-of-two for now.
        :::
      since: v0.5.14
      static: true
      tags: ["decorator"]
      params:
        - name: n
          type: int
          desc: threads per block / block dimension.
      returns:
        - type: Callable
---
