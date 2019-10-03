Contributor Guide
===============================================

Code Format
--------------------------------------------------------------------------
- We rely on `clang-format-4.0` for code format.
- Make sure to format your code before you commit, since we have not set up a githook for it.

Scoping
--------------------------------------------------------------------------


Naming
--------------------------------------------------------------------------
- Variable names should consist of lowercase words connected by underscores, e.g. ``density_field``.
- Class and struct names should consist of words with first letters captalized, e.g. ``MultigridPreconditioner``.
- Template classes should start with T, like ``TVector, TMatrix, TArray``;
   - Reserve the name without ``T`` for specialized classes, e.g. ``using Vector=TVector<real, dim>``.
- Macros should be capital start with ``TC``, such as ``TC_INFO``, ``TC_IMPLEMENTATION``.
   - We do not encourage the use of macro, though there are cases where macros are inevitable.
- Filenames should constist of lowercase words connected by underscores, e.g. ``parallel_reduction.cpp``.

File Organization
----------------------------------------------------------------------------
- Put in the `projects` folder


Object-Oriented Programming
-----------------------------------------------------------------------------

Common Patterns
-------------------------------------------------------------------------------


Casting
-------------------------------------------------------------------------------
- We allow the use of old-style C casting e.g. ``auto t = (int)x;``
    - Reason: ``static_cast<type>(variable)`` is too verbose.
- Think twice when you use ``reinterpret_cast``, ``const_cast``.
- Discussions on this in `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html#Casting>`_.


Do’s
-------------------------------------------------------------------------------
- Be considerate to your users (including yourself in the near future).
- Use ``auto`` for local variables when appropriate.
- Mark ``override`` and ``const`` when necessary.

Dont’s
--------------------------------------------------------------------------------
- C language legacies:
   -  ``printf`` (use ``fmtlib`` instead).
   -  ``new`` & ``free``. Use smart pointers (``std::unique_ptr, std::shared_ptr`` instead for ownership management).
   -  Unnecessary dependencies.
- Prefix member functions with ``m_`` or ``_``. Modern IDE can highlight members variables for you.
- Virtual function call in constructors/destructors.
- `C++ exceptions <https://google.github.io/styleguide/cppguide.html#Exceptions>`_
- ``NULL``, use ``nullptr`` instead.
- ``using namespace std;`` in headers global scope.
- ``typedef``. Use ``using`` instead.

Documentation
-------------------------------------------------------------------------------
- To build the documentation: ``ti doc`` or ``cd docs && sphinx-build -b html . build``.
