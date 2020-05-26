.. _external:

Interacting with external arrays
================================

**External arrays** refer to ``numpy.ndarray`` or ``torch.Tensor``.

Conversion between Taichi tensors and external arrays
-----------------------------------------------------

Use ``to_numpy``/``from_numpy``/``to_torch``/``from_torch``:

.. code-block:: python

  import taichi as ti
  import numpy as np

  ti.init()

  n = 4
  m = 7

  # Taichi tensors
  val = ti.var(ti.i32, shape=(n, m))
  vec = ti.Vector(3, dt=ti.i32, shape=(n, m))
  mat = ti.Matrix(3, 4, dt=ti.i32, shape=(n, m))

  # Scalar
  arr = np.ones(shape=(n, m), dtype=np.int32)

  val.from_numpy(arr)

  arr = val.to_numpy()

  # Vector
  arr = np.ones(shape=(n, m, 3), dtype=np.int32)

  vec.from_numpy(arr)

  arr = np.ones(shape=(n, m, 3, 1), dtype=np.int32)
  vec.from_numpy(arr)

  arr = vec.to_numpy()
  assert arr.shape == (n, m, 3, 1)

  arr = vec.to_numpy(as_vector=True)
  assert arr.shape == (n, m, 3)

  # Matrix
  arr = np.ones(shape=(n, m, 3, 4), dtype=np.int32)

  mat.from_numpy(arr)

  arr = mat.to_numpy()
  assert arr.shape == (n, m, 3, 4)


Using external arrays as Taichi kernel parameters
-------------------------------------------------

The type hint for external array parameters is ``ti.ext_arr()``. Please see the example below.
Note that struct-for's on external arrays are not supported.

.. code-block:: python

  import taichi as ti
  import numpy as np

  ti.init()

  n = 4
  m = 7

  val = ti.var(ti.i32, shape=(n, m))

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
