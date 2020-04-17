Internal designs (WIP)
======================

Vector type system
------------------


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
