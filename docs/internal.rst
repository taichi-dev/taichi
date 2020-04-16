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

In some cases, it is helpful gather certain quantitative information about internal events during
Taichi program execution. The ``Statistics`` class is designed for this purpose.

Usage:

::

    #include "taichi/util/statistics.h"

    taichi::stat.add("codegen_offloaded_tasks"); // add 1.0 to
    taichi::stat.add("codegen_statements", irpass::analysis::count_statements(this->ir));


Note the keys are `std::string`` and values are ``double``.

To print out all statistics in Python:

::

    ti.core.print_statistics()
