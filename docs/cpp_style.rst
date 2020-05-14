C++ Style
=========

Naming
------
- Variable names should consist of lowercase words connected by underscores, e.g. ``llvm_context``.
- Class and struct names should consist of words with first letters capitalized, e.g. ``CodegenLLVM``.
- Macros should be capital start with ``TI``, such as ``TI_INFO``, ``TI_IMPLEMENTATION``.

   - We do not encourage the use of macro, although there are cases where macros are inevitable.

- Filenames should consist of lowercase words connected by underscores, e.g. ``ir_printer.cpp``.

Dos
---
- Use ``auto`` for local variables when appropriate.
- Mark ``override`` and ``const`` when necessary.

Don'ts
------
- C language legacies:

   -  ``printf`` (Use ``fmtlib::print`` instead).
   -  ``new`` & ``free``. (Use smart pointers ``std::unique_ptr, std::shared_ptr`` instead for ownership management).
   -  ``#include <math.h>`` (Use ``#include <cmath>`` instead).

- Exceptions (we are on our way to remove all exceptions in Taichi).
- Prefix member functions with ``m_`` or ``_``.
- Virtual function call in constructors/destructors.
- ``NULL`` (use ``nullptr`` instead).
- ``using namespace std;`` in global scope.
- ``typedef`` (Use ``using`` instead).

Automatic code formatting
-------------------------
- Please run ``ti format``
