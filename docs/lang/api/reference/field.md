---
title: Field
docs:
  desc: |
    We provide interfaces to copy data between Taichi fields and NumPy arrays.
  functions:
    - name: to_numpy
      desc: Converts a field object to numpy ndarray.
      since: v0.5.14
      static: false
      tags: ["numpy"]
      params:
        - name: self
          type: ti.field | ti.Vector.field | ti.Matrix.field
          desc: The field.
      returns:
        - type: np.ndarray
          desc: The numpy array containing the current data in `x`.
    - name: from_numpy
      desc: Creates a field object from numpy ndarray.
      since: v0.5.14
      static: false
      tags: ["numpy"]
      params:
        - name: self
          type: ti.field | ti.Vector.field | ti.Matrix.field
          desc: The field.
        - name: array
          type: np.ndarray
      returns:
        - type: None
    - name: to_torch
      desc: Converts a field object to PyTorch Tensor.
      since: v0.5.14
      static: false
      tags: ["PyTorch"]
      params:
        - name: self
          type: ti.field | ti.Vector.field | ti.Matrix.field
          desc: The field.
        - name: device
          type: torch.device
          desc: The device where the PyTorch tensor is stored.
      returns:
        - type: torch.Tensor
          desc: The PyTorch tensor containing data in `x`.
    - name: from_torch
      desc: Creates a field object from PyTorch Tensor.
      since: v0.5.14
      static: false
      tags: ["PyTorch"]
      params:
        - name: self
          type: ti.field | ti.Vector.field | ti.Matrix.field
          desc: The field.
        - name: torch.Tensor
          type: The PyTorch tensor with data to initialize the field.
      returns:
        - type: None
---
