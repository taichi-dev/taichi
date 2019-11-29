Contributor Guide
===============================================

Naming
--------------------------------------------------------------------------
- Variable names should consist of lowercase words connected by underscores, e.g. ``density_field``.
- Class and struct names should consist of words with first letters capitalized, e.g. ``MultigridPreconditioner``.
- Macros should be capital start with ``TC``, such as ``TC_INFO``, ``TC_IMPLEMENTATION``.
   - We do not encourage the use of macro, though there are cases where macros are inevitable.
- Filenames should consist of lowercase words connected by underscores, e.g. ``parallel_reduction.cpp``.

Do’s
-------------------------------------------------------------------------------
- Use ``auto`` for local variables when appropriate.
- Mark ``override`` and ``const`` when necessary.

Dont’s
--------------------------------------------------------------------------------
- C language legacies:
   -  ``printf`` (use ``fmtlib::print`` instead).
   -  ``new`` & ``free``. Use smart pointers (``std::unique_ptr, std::shared_ptr`` instead for ownership management).
   -  Unnecessary dependencies.
- Prefix member functions with ``m_`` or ``_``.
- Virtual function call in constructors/destructors.
- `C++ exceptions <https://google.github.io/styleguide/cppguide.html#Exceptions>`_
- ``NULL``, use ``nullptr`` instead.
- ``using namespace std;`` in global scope.
- ``typedef``. Use ``using`` instead.

Documentation
-------------------------------------------------------------------------------
- To build the documentation: ``ti doc`` or ``cd docs && sphinx-build -b html . build``.
