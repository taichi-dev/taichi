Code style
==========

C++ style
---------

We generally follow `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_.

Naming
******
- Variable and function names should consist of lowercase words connected by underscores, e.g. ``llvm_context`` and ``get_current_kernel``.
  * Ideally private member variables should additionally end with ``_``, e.g. ``metal_compiled_structs_``.
- Class and struct names should consist of words with first letters capitalized, e.g. ``CodegenLLVM``.
- Macros should be capital start with ``TI``, such as ``TI_INFO``, ``TI_IMPLEMENTATION``.
  * We do not encourage the use of macro, although there are cases where macros are inevitable.
- Filenames should consist of lowercase words connected by underscores, e.g. ``ir_printer.cpp``.

Dos
***
- Use ``auto`` for local variables when appropriate.
- Mark ``override`` and ``const`` when necessary.

Don'ts
******
- C language legacies:

   -  ``printf`` (Use ``fmtlib::print`` instead).
   -  ``new`` and ``free``. (Use smart pointers ``std::unique_ptr, std::shared_ptr`` instead for ownership management).
   -  ``#include <math.h>`` (Use ``#include <cmath>`` instead).

- Exceptions (We are on our way to **remove** all C++ exception usages in Taichi).
- Prefix member functions with ``m_`` or ``_``.
- Virtual function call in constructors/destructors.
- ``NULL`` (Use ``nullptr`` instead).
- ``using namespace std;`` in the global scope.
- ``typedef`` (Use ``using`` instead).


Python style
------------

We generally follow `PEP8 <https://pep8.org>`_ for Python style.

Identation
**********
We always use **4 spaces** for indent. No tabs.

Naming
******
- Variable and function names should consist of lowercase words connected by underscores, e.g. ``num_args`` and ``adaptive_arch_select``.
  * Non-public APIs should additionally start with ``_``, e.g. ``_kernel_impl``. So that end-users won't get confused.
  * Ideally public APIs should be short and easy-to-memorize, unless it's harming readability, e.g. ``ti.activate`` instead of ``ti.activate_sparse_snode_at_index``.
  * Variable names like ``i`` or ``a`` should only occur in trivial places, otherwise is inacceptable.
  * Constant variables should be consisit of words with all latters capitalize, e.g. ``ti.GUI.PRESS``, ``N``.
- Class and struct names should consist of words with first letters capitalized, e.g. ``KernelTemplateMapper``.
  * When using abbreviations in CapWords, capitalize all the letters of the abbreviation, e.g. ``ASTTransformer`` instead of ``AstTransformer``.
- Filenames should consist of lowercase words connected by underscores, e.g. ``ast_checker.py``.

Dos
***
- Make good use of default arguments for simplicity.
- Use Python decorators for functions when appropriate.
- The operators tend to get scattered across different columns on the screen, and each operator is moved away from its operand and onto the previous line, e.g.:

.. code-block:: python

   income = (gross_wages
          + taxable_interest
          + (dividends - qualified_dividends)
          - ira_deduction
          - student_loan_interest)

Don'ts
******
- Mixed tabs and spaces. (Please always use 4 spaces for indent)
- Too long lines. (Please make use of local variables to break it down)
- Return ``None`` on failure. (Please raise an exception loudly when failure)
- Use a non-trivial value as inline default arguments:

.. code-block:: python

   def func(x, y=1):  # OK! Since 1 is literal constant so there won't be any problem.
      ...

   def func(x, y=sys.stdout):  # BAD! When sys.stdout is changed by other packages, it won't update when func() is called
      ...

   def func(x, y=None):  # GOOD! Instead, we use `None` as default argument, and switch to the real default argument within the function body for more flexibility
      if y is None:
        y = sys.stdout
      ...


Automatic code formatting
-------------------------

There are three ways to format your code.

1. Run ``ti format`` locally.
2. Click the link to format server in PR description.
2. Request `@taichi-gardener <https://github.com/taichi-gardener>`_ (a bot account) for review in your PR.
