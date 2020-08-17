.. _external:

Interacting with external arrays
================================

Although Taichi fields are mainly used in Taichi-scope, in some cases efficiently manipulating Taichi field data in Python-scope could also be helpful.

We provide various interfaces to copy the data between Taichi fields and external arrays.
The most typical case maybe copying between Tachi fields and Numpy arrays. Let's take a look at two examples below.

**Export data in Taichi fields to a NumPy array** via ``to_numpy()``.
This allows us to export computation results to other
Python packages that support NumPy, e.g. ``matplotlib``.

.. code-block:: python

   @ti.kernel
   def my_kernel():
      for i in x:
         x[i] = i * 2

   x = ti.field(ti.f32, 4)
   my_kernel()
   x_np = x.to_numpy()
   print(x_np)  # np.array([0, 2, 4, 6])


**Import data from NumPy array to Taichi fields** via ``from_numpy()``.
This allows people to initialize Taichi fields via NumPy arrays. E.g.,

.. code-block:: python

   x = ti.field(ti.f32, 4)
   x_np = np.array([1, 7, 3, 5])
   x.from_numpy(x_np)
   print(x[0])  # 1
   print(x[1])  # 7
   print(x[2])  # 3
   print(x[3])  # 5


API reference
-------------

We provide interfaces to copy data between Taichi field and  **external arrays**.
External arrays refers to NumPy arrays or PyTorch tensors.

We suggest common users to start with NumPy arrays.

Interacting with NumPy
**********************

We provide interfaces to copy data between Taichi fields and NumPy arrays.

.. function:: field.to_numpy()

   :parameter field: (ti.field, ti.Vector.field or ti.Matrix.field) The field

   :return: (np.array) The numpy array containing the current data in ``x``.

.. function:: field.from_numpy(array)

   :parameter field: (ti.field, ti.Vector.field or ti.Matrix.field) The field

   :parameter array: (np.array) The numpy array containing data to initialize the field


Interacting with PyTorch
************************

We also provide interfaces to copy data directly between Taichi fields and PyTorch tensors.

.. function:: field.to_torch(device = None)

   :parameter field: (ti.field, ti.Vector.field or ti.Matrix.field) The field
   :parameter device: (torch.device) the device where the PyTorch tensor is stored

   :return: (torch.Tensor) The PyTorch tensor containing data in ``x``.

.. function:: field.from_torch(tensor)

   :parameter field: (ti.field, ti.Vector.field or ti.Matrix.field) The field

   :parameter tensor: (torch.Tensor) The PyTorch tensor with data to initialize the field


External array shapes
---------------------

Shapes of Taichi fields (see :ref:`scalar_tensor`) and those of corresponding NumPy arrays are closely connected via the following rules:

- For scalar fields, **the shape of NumPy array is exactly the same as the Taichi field**:

.. code-block:: python

   field = ti.field(ti.i32, shape=(233, 666))
   field.shape  # (233, 666)

   array = field.to_numpy()
   array.shape  # (233, 666)

   field.from_numpy(array)  # the input array must be of shape (233, 666)


- For vector fields, if the vector is ``n``-D, then **the shape of NumPy array should be** ``(*field_shape, vector_n)``:

.. code-block:: python

   field = ti.Vector.field(3, ti.i32, shape=(233, 666))
   field.shape  # (233, 666)
   field.n      # 3

   array = field.to_numpy()
   array.shape  # (233, 666, 3)

   field.from_numpy(array)  # the input array must be of shape (233, 666, 3)


- For matrix fields, if the matrix is ``n*m``, then **the shape of NumPy array should be** ``(*field_shape, matrix_n, matrix_m)``:

.. code-block:: python

   field = ti.Matrix.field(3, 4, ti.i32, shape=(233, 666))
   field.shape  # (233, 666)
   field.n      # 3
   field.m      # 4

   array = field.to_numpy()
   array.shape  # (233, 666, 3, 4)

   field.from_numpy(array)  # the input array must be of shape (233, 666, 3, 4)


Using external arrays as Taichi kernel arguments
------------------------------------------------

Use the type hint ``ti.ext_arr()`` for passing external arrays as kernel
arguments. For example:

.. code-block:: python

  import taichi as ti
  import numpy as np

  ti.init()

  n = 4
  m = 7

  val = ti.field(ti.i32, shape=(n, m))

  @ti.kernel
  def test_numpy(arr: ti.ext_arr()):
    for i in range(n):
      for j in range(m):
        arr[i, j] += i + j

  a = np.empty(shape=(n, m), dtype=np.int32)

  for i in range(n):
    for j in range(m):
      a[i, j] = i * j

  test_numpy(a)

  for i in range(n):
    for j in range(m):
      assert a[i, j] == i * j + i + j

.. note::

   Struct-for's are not supported on external arrays.
