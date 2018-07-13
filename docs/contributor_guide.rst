Contributor Guide
===============================================

Code Format
--------------------------------------------------------------------------
- We rely on `clang-format-4.0` for code format.
- Make sure you format your code before you commit since we have not set up a githook for it.

Scoping
--------------------------------------------------------------------------


Naming
--------------------------------------------------------------------------
- Variables should be lowercase words connected by underscores, like ``density_field`` 
- Classes and struct should start with a capital letter  
- Template classes should start with T, like ``TVector, TMatrix, TArray``
   - Advantages: ``using Vector=Vector<int, dim>``

- Macros should start with ``TC``, such as ``TC_INFO``.  

- Filenames

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
    - Reason: static_cast is too verbose
- Think twice when you use ``reinterpret_cast``, ``const_cast``
- Discussions on this in `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html#Casting>`_


Do’s
-------------------------------------------------------------------------------
- Be considerate to your users (this includes yourself in the near future)
- Use ``auto`` for local variables when appropriate
- Mark ``override`` and ``const`` when necessary



Dont’s
--------------------------------------------------------------------------------
- C language legacies:
   -  ``printf`` (use ``fmtlib`` instead)
   -  ``new`` & ``free``. Use smart pointers (``std::unique_ptr, std::shared_ptr`` instead for ownership management)
   -  Unnecessary dependencies

- Prefix member functions with ``m_`` or ``_``. Modern IDE can highlight members variables for you.

- Virtual function call in constructors/destructors.

- `C++ exceptions <https://google.github.io/styleguide/cppguide.html#Exceptions>`_

- ``NULL``, use ``nullptr`` instead.
 
- ``using namespace std;`` in headers global scope.

- ``typedef``. Use ``using`` instead.

Documentation
-------------------------------------------------------------------------------
```cd dos && sphinx-build -b html . build``
