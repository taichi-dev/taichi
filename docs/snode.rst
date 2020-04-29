.. _snode:

Struct node
===========

Struct nodes, or SNode, each represents the layer of structure of *layout*.
TODO: describe the idea better here.


.. function:: snode.place(x, ...)

    :parameter snode: (SNode) where to place
    :parameter x: (tensor) tensor(s) to be placed
    :return: (SNode) the ``snode`` itself unchaged


.. function:: ti.root

    ``ti.root`` is a kind of SNode, The root SNode, stands for 0-D tensor.

    This places a 0-D tensor:

    ::

        ti.root.place(x)


Indices
-------

.. function:: ti.i
.. function:: ti.j
.. function:: ti.k
.. function:: ti.ij
.. function:: ti.ijk
.. function:: ti.ijkl
.. function:: ti.indices(a, b, ...)

TODO: complete equivalent descs here


Data layouts
------------


.. function:: snode.dense(indices, shape)

    :parameter snode: (SNode) parent SNode where the child derived from
    :parameter indices: (Index or Indices) indices used for this node
    :parameter shape: (scalar or tuple) shape the tensor of vectors
    :return: (SNode) the derived child SNode

    This places a 1-D tensor of size ``3``:

    ::

        ti.root.dense(ti.i, 3).place(x)

    This places a 1-D tensor of shape ``(3, 4)``:

    ::

        ti.root.dense(ti.ij, (3, 4)).place(x)

    .. note::

        If ``shape`` is scalar instead of tuple, and there is more than one indices, then the ``shape`` will be automatically expanded to fit the indices, e.g.:

        ::

            snode.dense(ti.ijk, 3)

        will be translated into:

        ::

            snode.dense(ti.ijk, (3, 3, 3))


.. function:: snode.dynamic(index, size, chunk_size = None)

    :parameter snode: (SNode) parent SNode where the child derived from
    :parameter index: (Index) index used for this node
    :parameter shape: (scalar) the initial value of dynamic size
    :return: (SNode) the derived child SNode

    The size of dynamic SNodes can be extended in runtime, see functions below.

    This places a 1-D dynamic tensor of initial size ``3``:

    ::

        ti.root.dynamic(ti.i, 3).place(x)

    This places a 2D tensor of shape ``(3, 4)``:

    ::

        ti.root.dense(ti.ij, (3, 4)).place(x)


.. function:: ti.length(snode)

    :parameter snode: (SNode, dynamic)
    :return: (scalar) current size of the dynamic SNode


.. function:: ti.append(snode, indices, val)

    :parameter snode: (SNode, dynamic)
    :parameter indices: (scalar or tuple) indices within SNode
    :parameter val: (depends on SNode data type) value to store

    ASK(yuanming-hu): how is this used exactly??
