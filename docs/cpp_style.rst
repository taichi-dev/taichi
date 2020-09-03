Code style
==========

C++ style
---------

We generally follow `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_.

For example:

.. code-block:: cpp

    namespace mynamespace {

        class MyClass {
            ...
        };

        void my_function(MyClass *mc, int value) {
            mc->some_value = value;
            if (mc->some_ptr != nullptr) {
                mc->some_method();  // some comments
            } else if (mc->some_other_ptr != nullptr) {
                // TODO: some todo notes
                mc->some_other_method();
            } else {
                TI_NOT_IMPLEMENTED
            }
        }

    }  // namespace mynamespace

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

   *  ``printf`` (Use ``fmtlib::print`` instead).
   *  ``new`` and ``free``. (Use smart pointers ``std::unique_ptr, std::shared_ptr`` instead for ownership management).
   *  ``#include <math.h>`` (Use ``#include <cmath>`` instead).

- Exceptions (We are on our way to **remove** all C++ exception usages in Taichi).
- Prefix member functions with ``m_`` or ``_``.
- Virtual function call in constructors/destructors.
- ``NULL`` (Use ``nullptr`` instead).
- ``using namespace std;`` in the global scope.
- ``typedef`` (Use ``using`` instead).
- Misuse of ``&`` (references).


Python style
------------

We generally follow `PEP8 <https://pep8.org>`_ for Python style.

For example:

.. code-block:: python

    import numpy as np
    from .lang import my_decorator

    class MyClass:
        ...

    @my_decorator
    def my_function(mc, value=None):
        if value is None:
            value = np.array([2, 3])

        mc.some_value = value
        if mc.some_obj is not None:
            mc.some_method()  # some comments
        elif mc.some_other_obj is not None:
            # TODO: some todo notes
            mc.some_other_method()
        else:
            raise NotImplementedError('Some error messages')

Identation
**********
We always use **4 spaces** for indent. No tabs.

Naming
******
- Variable and function names should follow snake_case, e.g. ``num_args`` and ``adaptive_arch_select``.
  * Non-public APIs should additionally start with ``_``, e.g. ``_kernel_impl``. So that end-users won't get confused.
  * Ideally public APIs should be short and easy-to-memorize, as long as it does no harm to readability, e.g. ``ti.activate`` instead of ``ti.activate_sparse_snode_at_index``.
  * Single-character variable names like ``i`` or ``a`` should only occur in trivial places, otherwise they are inacceptable.
  * Constant variables should consist of words with all-uppercase letters, e.g. ``ti.GUI.PRESS``, ``N``.
- Class and struct names should follow PascalCase, e.g. ``KernelTemplateMapper``.
  * When using abbreviations in CapWords, capitalize all the letters of the abbreviation, e.g. ``ASTTransformer`` instead of ``AstTransformer``.
- Module names should consist of lowercase words connected by underscores (snake_case), e.g. ``ast_checker.py``.

Dos
***
- Make good use of default arguments for simplicity.
- Use Python decorators for functions when appropriate.
- Use relative import, e.g. ``from .lang import Matrix`` instead of ``from taichi.lang import Matrix``.
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
- Lines longer than 88 characters. (Please make use of local variables to break it down)
- Return ``None`` on failure. (Please raise an exception loudly when failure)
- Use ``from xxx import *`` without specifying ``__all__`` in module ``xxx``.
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
