.. _snode:

Structural nodes
================

After writing the computation code, the user needs to specify the internal data structure hierarchy. Specifying a data structure includes choices at both the macro level, dictating how the data structure components nest with each other and the way they represent sparsity, and the micro level, dictating how data are grouped together (e.g. structure of arrays vs. array of structures).
Our language provides *structural nodes* to compose the hierarchy and particular properties. These constructs and their semantics are listed below:

* dense: A fixed-length contiguous array.

* dynamic: Variable-length array, with a predefined maximum length. It serves the role of ``std::vector``, and can be used to maintain objects (e.g. particles) contained in a block.

* bitmasked: Use a mask to maintain sparsity infomation, one bit per child.

* pointer: Store pointers instead of the whole structure to save memory and maintain sparsity.

* hash: Use a hash table to maintain the mapping from active coordinates to data addresses in memory. Suitable for high sparsity.

See :ref:`layout` for more details about data layout.


.. function:: snode.place(x, ...)

    :parameter snode: (SNode) where to place
    :parameter x: (tensor) tensor(s) to be placed
    :return: (SNode) the ``snode`` itself


.. function:: ti.root

    ``ti.root`` is a kind of structural node, The root node, stands for 0-D tensor.

    This places two 0-D tensors named ``x`` and ``y``:

    ::

        ti.root.place(x, y)


Node types
----------


.. function:: snode.dense(indices, shape)

    :parameter snode: (SNode) parent node where the child is derived from
    :parameter indices: (Index or Indices) indices used for this node
    :parameter shape: (scalar or tuple) shape the tensor of vectors
    :return: (SNode) the derived child node

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
.. function:: snode.hash
.. function:: snode.bitmasked
.. function:: snode.pointer

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

TODO: complete equivalent descs here
