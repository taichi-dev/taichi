.. _snode:

Structural nodes (SNodes)
=========================

After writing the computation code, the user needs to specify the internal data structure hierarchy. Specifying a data structure includes choices at both the macro level, dictating how the data structure components nest with each other and the way they represent sparsity, and the micro level, dictating how data are grouped together (e.g. structure of arrays vs. array of structures).
Our language provides *structural nodes (SNodes)* to compose the hierarchy and particular properties. These constructs and their semantics are listed below:

* dense: A fixed-length contiguous array.

* bitmasked: Use a mask to maintain sparsity information, one bit per child.

* pointer: Store pointers instead of the whole structure to save memory and maintain sparsity.

* hash: Use a hash table to maintain the mapping from active coordinates to data addresses in memory. Suitable for high sparsity.

* dynamic: Variable-length array, with a predefined maximum length. It serves the role of ``std::vector`` in C++ or ``list`` in Python, and can be used to maintain objects (e.g. particles) contained in a block.

See :ref:`layout` for more details about data layout. ``ti.root`` is the root node of the data structure.

.. function:: snode.place(x, ...)

    :parameter snode: (SNode) where to place
    :parameter x: (tensor) tensor(s) to be placed
    :return: (SNode) the ``snode`` itself

.. function:: snode.get_shape(index)

    :parameter snode: (SNode)
    :parameter index: alone which axis? (0 for ``i`` and 1 for ``j``)
    :return: (scalar) the size of tensor alone that axis

    Equivalent to snode.shape[i]

    ::

        ti.root.dense(ti.ijk, (3, 5, 4)).place(x)
        x.get_shape(0)  # 3
        x.get_shape(1)  # 5
        x.get_shape(2)  # 4

.. function:: snode.dim()

    :parameter snode: (SNode)
    :return: (scalar) the dimension of node

    Equivalent to `len(snode.shape)`.

    ::

        ti.root.dense(ti.ijk, (8, 9, 10)).place(x)
        x.dim()  # 3

    TODO: add tensor.dim(), and add see also ref here

.. function:: snode.parent()

    :parameter snode: (SNode)
    :return: (SNode) the parent node of `snode`

    ::

        blk1 = ti.root.dense(ti.i, 8)
        blk2 = blk1.dense(ti.j, 4)
        blk3 = blk2.bitmasked(ti.k, 6)
        blk1.parent()  # ti.root
        blk2.parent()  # blk1
        blk3.parent()  # blk2

    TODO: add tensor.parent(), and add see also ref here

    The following code places two 0-D tensors named ``x`` and ``y``:

    ::

        x = ti.var(dt=ti.i32)
        y = ti.var(dt=ti.f32)
        ti.root.place(x, y)


Node types
----------


.. function:: snode.dense(indices, shape)

    :parameter snode: (SNode) parent node where the child is derived from
    :parameter indices: (Index or Indices) indices used for this node
    :parameter shape: (scalar or tuple) shape the tensor of vectors
    :return: (SNode) the derived child node

    The following code places a 1-D tensor of size ``3``:

    ::

        x = ti.var(dt=ti.i32)
        ti.root.dense(ti.i, 3).place(x)

    The following code places a 2-D tensor of shape ``(3, 4)``:

    ::

        x = ti.var(dt=ti.i32)
        ti.root.dense(ti.ij, (3, 4)).place(x)

    .. note::

        If ``shape`` is a scalar and there are multiple indices, then ``shape`` will
        be automatically expanded to fit the number of indices. For example,

        ::

            snode.dense(ti.ijk, 3)

        is equivalent to

        ::

            snode.dense(ti.ijk, (3, 3, 3))


.. function:: snode.dynamic(index, size, chunk_size = None)

    :parameter snode: (SNode) parent node where the child is derived from
    :parameter index: (Index) the ``dynamic`` node indices
    :parameter size: (scalar) the maximum size of the dynamic node
    :parameter chunk_size: (optional, scalar) the number of elements in each dynamic memory allocation chunk
    :return: (SNode) the derived child node

    ``dynamic`` nodes acts like ``std::vector`` in C++ or ``list`` in Python.
    Taichi's dynamic memory allocation system allocates its memory on the fly.

    The following places a 1-D dynamic tensor of maximum size ``16``:

    ::

        ti.root.dynamic(ti.i, 16).place(x)


.. function:: ti.length(snode, indices)

    :parameter snode: (SNode, dynamic)
    :parameter indices: (scalar or tuple of scalars) the ``dynamic`` node indices
    :return: (scalar) the current size of the dynamic node


.. function:: ti.append(snode, indices, val)

    :parameter snode: (SNode, dynamic)
    :parameter indices: (scalar or tuple of scalars) the ``dynamic`` node indices
    :parameter val: (depends on SNode data type) value to store
    :return: (``int32``) the size of the dynamic node, before appending

    Inserts ``val`` into the ``dynamic`` node with indices ``indices``.


.. function:: snode.bitmasked
.. function:: snode.pointer
.. function:: snode.hash

    TODO: add descriptions here


Indices
-------

.. function:: ti.i
.. function:: ti.j
.. function:: ti.k
.. function:: ti.ij
.. function:: ti.ijk
.. function:: ti.ijkl
.. function:: ti.indices(a, b, ...)

(TODO)
