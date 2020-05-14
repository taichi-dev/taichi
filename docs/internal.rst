Internal designs (WIP)
======================


Intermediate representation
---------------------------
Use ``ti.init(print_ir=True)`` to print IR on the console.


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

  * For example, indexing is always needed when accessing elements in Taichi tensors, even if the tensor is 0D. Use ``x[None] = 123`` to set the value in ``x`` if ``x`` is 0D. This is because ``x = 123`` will set ``x`` itself (instead of its containing value) to be the constant ``123`` in python syntax, and, unfortunately, we cannot modify this behavior.

* Python has relatively low performance. This can cause a performance issue when initializing large Taichi tensors with pure python scripts. A Taichi kernel should be used to initialize a huge tensor.
