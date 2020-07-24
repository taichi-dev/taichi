Internal designs (WIP)
======================

Intermediate representation
---------------------------
Use ``ti.init(print_ir=True)`` to print IR on the console.


List generation (WIP)
---------------------

Struct-fors in Taichi loop over all active elements of a (sparse) data structure **in parallel**.
Evenly distributing work onto processor cores is challenging on sparse data structures:
naively splitting an irregular tree into pieces can easily lead to
partitions with drastically different numbers of leaf elements.

Our strategy is to generate lists of active SNode elements layer by layer.
The list generation computation happens on the same device as normal computation kernels,
depending on the ``arch`` argument when the user calls ``ti.init``.

List generations flatten the data structure leaf elements into a 1D dense array, circumventing the irregularity of
incomplete trees. Then we can simply invoke a regular **parallel for** over the list.

For example,

.. code-block:: python

    # misc/listgen_demo.py

    import taichi as ti

    ti.init(print_ir=True)

    x = ti.field(ti.i32)
    ti.root.dense(ti.i, 4).bitmasked(ti.i, 4).place(x)

    @ti.kernel
    def func():
        for i in x:
            print(i)

    func()


gives you the following IR:

.. code-block:: none

  $0 = offloaded clear_list S1dense
  $1 = offloaded listgen S0root->S1dense
  $2 = offloaded clear_list S2bitmasked
  $3 = offloaded listgen S1dense->S2bitmasked
  $4 = offloaded struct_for(S2bitmasked) block_dim=0 {
    <i32 x1> $5 = loop index 0
    print i, $5
  }


Note that ``func`` leads to two list generations:

 - (Tasks ``$0`` and ``$1``) based on the list of ``root`` node (``S0``), generate the list of the ``dense`` nodes (``S1``);
 - (Tasks ``$2`` and ``$3``) based on the list of ``dense`` nodes (``S1``), generate the list of ``bitmasked`` nodes (``S2``).

The list of ``root`` node always has exactly one element (instance), so we never clear or re-generate this list.

.. note::

    The list of ``place`` (leaf) nodes (e.g., ``S3`` in this example) is never generated.
    Instead, we simply loop over the list of their parent nodes, and for each parent node we
    enumerate the ``place`` nodes on-the-fly (without actually generating a list).

    The motivation for this design is to amortize list generation overhead.
    Generating one list element per leaf node (``place`` SNode) element is too expensive, likely much
    more expensive than the essential computation happening on the leaf element.
    Therefore we only generate their parent element list, so that the list generation cost is
    amortized over multiple child elements of a second-to-last-level SNode element.

    In the example above, although we have ``16`` instances of ``x``,
    we only generate a list of ``4`` ``bitmasked`` nodes (and ``1`` ``dense`` node).


Code generation
---------------


Statistics
----------

In some cases, it is helpful to gather certain quantitative information about internal events during
Taichi program execution. The ``Statistics`` class is designed for this purpose.

Usage:

.. code-block:: C++

    #include "taichi/util/statistics.h"

    // add 1.0 to counter "codegen_offloaded_tasks"
    taichi::stat.add("codegen_offloaded_tasks");

    // add the number of statements in "ir" to counter "codegen_statements"
    taichi::stat.add("codegen_statements", irpass::analysis::count_statements(this->ir));


Note the keys are ``std::string`` and values are ``double``.

To print out all statistics in Python:

.. code-block:: Python

    ti.core.print_stat()


Why Python frontend
-------------------

Embedding Taichi in ``python`` has the following advantages:

* Easy to learn. Taichi has a very similar syntax to Python.
* Easy to run. No ahead-of-time compilation is needed.
* This design allows people to reuse existing python infrastructure:

  * IDEs. A python IDE mostly works for Taichi with syntax highlighting, syntax checking, and autocomplete.
  * Package manager (pip). A developed Taichi application and be easily submitted to ``PyPI`` and others can easily set it up with ``pip``.
  * Existing packages. Interacting with other python components (e.g. ``matplotlib`` and ``numpy``) is just trivial.

* The built-in AST manipulation tools in ``python`` allow us to do magical things, as long as the kernel body can be parsed by the Python parser.

However, this design has drawbacks as well:

* Taichi kernels must parse-able by Python parsers. This means Taichi syntax cannot go beyond Python syntax.

  * For example, indexing is always needed when accessing elements in Taichi fields, even if the fields is 0D. Use ``x[None] = 123`` to set the value in ``x`` if ``x`` is 0D. This is because ``x = 123`` will set ``x`` itself (instead of its containing value) to be the constant ``123`` in python syntax, and, unfortunately, we cannot modify this behavior.

* Python has relatively low performance. This can cause a performance issue when initializing large Taichi fields with pure python scripts. A Taichi kernel should be used to initialize a huge fields.


Virtual indices v.s. physical indices
-------------------------------------

In Taichi, *virtual indices* are used to locate elements in fields, and *physical indices*
are used to specify data layouts in memory.

For example,

 - In ``a[i, j, k]``, ``i``, ``j``, and ``k`` are **virtual** indices.
 - In ``for i, j in x:``, ``i`` and ``j`` are **virtual** indices.
 - ``ti.i, ti.j, ti.k, ti.l, ...`` are **physical** indices.
 - In struct-for statements, ``LoopIndexStmt::index`` is a **physical** index.

The mapping between virtual indices and physical indices for each ``SNode`` is
stored in ``SNode::physical_index_position``.
I.e.,  ``physical_index_position[i]`` answers the question: **which physical index does the i-th virtual index**
correspond to?

Each ``SNode`` can have a different virtual-to-physical mapping. ``physical_index_position[i] == -1``
means the ``i``-th virtual index does not corrspond to any physical index in this ``SNode``.

``SNode`` s in handy dense fields (i.e., ``a = ti.field(ti.i32, shape=(128, 256, 512))``)
have **trivial** virtual-to-physical mapping, e.g. ``physical_index_position[i] = i``.

However, more complex data layouts, such as column-major 2D fields can lead to ``SNodes`` with
``physical_index_position[0] = 1`` and ``physical_index_position[1] = 0``.

.. code-block:: python

    a = ti.field(ti.f32, shape=(128, 32, 8))

    b = ti.field(ti.f32)
    ti.root.dense(ti.j, 32).dense(ti.i, 16).place(b)

    ti.get_runtime().materialize()

    mapping_a = a.snode().physical_index_position()

    assert mapping_a == {0: 0, 1: 1, 2: 2}

    mapping_b = b.snode().physical_index_position()

    assert mapping_b == {0: 1, 1: 0}
    # Note that b is column-major:
    # the virtual first index exposed to the user comes second in memory layout.

Taichi supports up to 8 (``constexpr int taichi_max_num_indices = 8``) virtual indices and physical indices.
