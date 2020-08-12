.. _snode:

Structural nodes (SNodes)
=========================

After writing the computation code, the user needs to specify the internal data structure hierarchy. Specifying a data structure includes choices at both the macro level, dictating how the data structure components nest with each other and the way they represent sparsity, and the micro level, dictating how data are grouped together (e.g. structure of arrays vs. array of structures).
Taichi provides *Structural Nodes (SNodes)* to compose the hierarchy and particular properties. These constructs and their semantics are listed below:

* dense: A fixed-length contiguous array.

* bitmasked: This is similar to dense, but it also uses a mask to maintain sparsity information, one bit per child.

* pointer: Store pointers instead of the whole structure to save memory and maintain sparsity.

* dynamic: Variable-length array, with a predefined maximum length. It serves the role of ``std::vector`` in C++ or ``list`` in Python, and can be used to maintain objects (e.g. particles) contained in a block.


See :ref:`layout` for more details. ``ti.root`` is the root node of the data structure.

.. function:: snode.place(x, ...)

    :parameter snode: (SNode) where to place
    :parameter a: (ti.field) field(s) to be placed
    :return: (SNode) the ``snode`` itself

    The following code places two 0-D fields named ``x`` and ``y``:

    ::

        x = ti.field(dtype=ti.i32)
        y = ti.field(dtype=ti.f32)
        ti.root.place(x, y)
        assert x.snode.parent == y.snode.parent


.. function:: field.shape

    :parameter a: (ti.field)
    :return: (tuple of integers) the shape of field

    Equivalent to ``field.snode.shape``.

    For example,

    ::

        ti.root.dense(ti.ijk, (3, 5, 4)).place(x)
        x.shape  # returns (3, 5, 4)


.. function:: field.snode

    :parameter a: (ti.field)
    :return: (SNode) the structual node where ``field`` is placed

    ::

        x = ti.field(dtype=ti.i32)
        y = ti.field(dtype=ti.f32)
        blk1 = ti.root.dense(ti.i, 4)
        blk1.place(x, y)
        assert x.snode == blk1


.. function:: snode.shape

    :parameter snode: (SNode)
    :return: (tuple) the size of node along that axis

    ::

        blk1 = ti.root
        blk2 = blk1.dense(ti.i,  3)
        blk3 = blk2.dense(ti.jk, (5, 2))
        blk4 = blk3.dense(ti.k,  2)
        blk1.shape  # ()
        blk2.shape  # (3, )
        blk3.shape  # (3, 5, 2)
        blk4.shape  # (3, 5, 4)


.. function:: snode.parent(n = 1)

    :parameter snode: (SNode)
    :parameter n: (optional, scalar) the number of steps, i.e. ``n=1`` for parent, ``n=2`` grandparent, etc.
    :return: (SNode) the parent node of ``snode``

    ::

        blk1 = ti.root.dense(ti.i, 8)
        blk2 = blk1.dense(ti.j, 4)
        blk3 = blk2.bitmasked(ti.k, 6)
        blk1.parent()  # ti.root
        blk2.parent()  # blk1
        blk3.parent()  # blk2
        blk3.parent(1) # blk2
        blk3.parent(2) # blk1
        blk3.parent(3) # ti.root
        blk3.parent(4) # None


Node types
----------


.. function:: snode.dense(indices, shape)

    :parameter snode: (SNode) parent node where the child is derived from
    :parameter indices: (Index or Indices) indices used for this node
    :parameter shape: (scalar or tuple) shape of the field
    :return: (SNode) the derived child node

    The following code places a 1-D field of size ``3``:

    ::

        x = ti.field(dtype=ti.i32)
        ti.root.dense(ti.i, 3).place(x)

    The following code places a 2-D field of shape ``(3, 4)``:

    ::

        x = ti.field(dtype=ti.i32)
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

    The following places a 1-D dynamic field of maximum size ``16``:

    ::

        ti.root.dynamic(ti.i, 16).place(x)



.. function:: snode.bitmasked
.. function:: snode.pointer
.. function:: snode.hash

    TODO: add descriptions here

.. _dynamic:

Working with ``dynamic`` SNodes
-------------------------------

.. function:: ti.length(snode, indices)

    :parameter snode: (SNode, dynamic)
    :parameter indices: (scalar or tuple of scalars) the ``dynamic`` node indices
    :return: (int32) the current size of the dynamic node


.. function:: ti.append(snode, indices, val)

    :parameter snode: (SNode, dynamic)
    :parameter indices: (scalar or tuple of scalars) the ``dynamic`` node indices
    :parameter val: (depends on SNode data type) value to store
    :return: (int32) the size of the dynamic node, before appending

    Inserts ``val`` into the ``dynamic`` node with indices ``indices``.


Taichi fields like powers of two
--------------------------------

Non-power-of-two field dimensions are promoted into powers of two and thus these fields will occupy more virtual address space.
For example, a (dense) field of size ``(18, 65)`` will be materialized as ``(32, 128)``.


Indices
-------

.. attribute:: ti.i
.. attribute:: ti.j
.. attribute:: ti.k
.. attribute:: ti.ij
.. attribute:: ti.ji
.. attribute:: ti.jk
.. attribute:: ti.kj
.. attribute:: ti.ik
.. attribute:: ti.ki
.. attribute:: ti.ijk
.. attribute:: ti.ijkl
.. function:: ti.indices(a, b, ...)

(TODO)
