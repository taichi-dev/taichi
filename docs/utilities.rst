Developer utilities
===================

This section provides a detailed description of some commonly used utilities for Taichi developers.

Logging
-------

Taichi uses `spdlog <https://github.com/gabime/spdlog>`_ as its logging system.
Logs can have different levels, from low to high, they are:

.. code-block:: none

    trace
    debug
    info
    warn
    error

The higher the level is, the more critical the message is.

The default logging level is ``info``. You may override the default logging level by:

1. Setting the environment variable like ``export TI_LOG_LEVEL=warn``.
2. Setting the log level from Python side: ``ti.set_logging_level(ti.WARN)``.

In Python, you may write logs using the ``ti.*`` interface:

.. code-block:: python

    # Python
    ti.trace("Hello world!")
    ti.debug("Hello world!")
    ti.info("Hello world!")
    ti.warn("Hello world!")
    ti.error("Hello world!")

In C++, you may write logs using the ``TI_*`` interface:

.. code-block:: cpp

    // C++
    TI_TRACE("Hello world!");
    TI_DEBUG("Hello world!");
    TI_INFO("Hello world!");
    TI_WARN("Hello world!");
    TI_ERROR("Hello world!");

If one raises a message of the level ``error``, Taichi will be **terminated** immediately
and result in a ``RuntimeError`` on Python side.

.. code-block:: cpp

  int func(void *p) {
    if (p == nullptr)
      TI_ERROR("The pointer cannot be null!");

    // will not reach here if p == nullptr
    do_something(p);
  }

.. note::

  For people from Linux kernels, ``TI_ERROR`` is just ``panic``.


You may also simplify the above code by using ``TI_ASSERT``:

.. code-block:: cpp

  int func(void *p) {
    TI_ASSERT_INFO(p != nullptr, "The pointer cannot be null!");
    // or
    // TI_ASSERT(p != nullptr);

    // will not reach here if p == nullptr
    do_something(p);
  }

.. _regress:

Benchmarking and regression tests
---------------------------------

* Run ``ti benchmark`` to run tests in benchmark mode. This will record the performance of ``ti test``, and save it in ``benchmarks/output``.

* Run ``ti regression`` to show the difference between the previous result in ``benchmarks/baseline``. And you can see if the performance is increasing or decreasing after your commits. This is really helpful when your work is related to IR optimizations.

* Run ``ti baseline`` to save the benchmark result to ``benchmarks/baseline`` for future comparison, this may be executed on performance-related PRs, before they are merged into master.

For example, this is part of the output by ``ti regression`` after enabling constant folding optimization pass:

.. code-block:: none

    linalg__________________polar_decomp______________________________
    codegen_offloaded_tasks                       37 ->    39    +5.4%
    codegen_statements                          3179 ->  3162    -0.5%
    codegen_kernel_statements                   2819 ->  2788    -1.1%
    codegen_evaluator_statements                   0 ->    14    +inf%

    linalg__________________init_matrix_from_vectors__________________
    codegen_offloaded_tasks                       37 ->    39    +5.4%
    codegen_statements                          3180 ->  3163    -0.5%
    codegen_kernel_statements                   2820 ->  2789    -1.1%
    codegen_evaluator_statements                   0 ->    14    +inf%

.. note::

    Currently ``ti benchmark`` only supports benchmarking number-of-statements, no time benchmarking is included since it depends on hardware performance and therefore hard to compare if the baseline is from another machine.
    We are to purchase a fixed-performance machine as a time benchmark server at some point.
    Discussion at: https://github.com/taichi-dev/taichi/issue/948


The suggested workflow for the performance-related PR author to run the regression tests is:

* Run ``ti benchmark && ti baseline`` in ``master`` to save the current performance as a baseline.

* Run ``git checkout -b your-branch-name``.

* Do works on the issue, stage 1.

* Run ``ti benchmark && ti regression`` to obtain the result.

* (If result BAD) Do further improvements, until the result is satisfying.

* (If result OK) Run ``ti baseline`` to save stage 1 performance as a baseline.

* Go forward to stage 2, 3, ..., and the same workflow is applied.


(Linux only) Trigger ``gdb`` when programs crash
------------------------------------------------

.. code-block:: none

  # Python
  ti.set_gdb_trigger(True)

  // C++
  CoreState::set_trigger_gdb_when_crash(true);

  # Shell
  export TI_GDB_TRIGGER=1


.. note::

  **Quickly pinpointing segmentation faults/assertion failures using** ``gdb``:
  When Taichi crashes, ``gdb`` will be triggered and attach to the current thread.
  You might be prompt to enter sudo password required for gdb thread attaching.
  After entering ``gdb``, check the stack backtrace with command ``bt`` (``backtrace``),
  then find the line of code triggering the error.


.. _coverage:

Code coverage
-------------

To ensure that our tests covered every situation, we need to have **coverage report**.
That is, to detect how many percents of code lines in is executed in test.

- Generally, the higher the coverage percentage is, the stronger our tests are.
- When making a PR, we want to **ensure that it comes with corresponding tests**. Or code coverage will decrease.
- Code coverage statuses are visible at `Codecov <https://codecov.io/gh/taichi-dev/taichi>`_.
- Currently, Taichi is only set up for Python code coverage report, not for C++ yet.

.. code-block:: bash

    ti test -C       # run tests and save results to .coverage
    coverage report  # generate a coverage report on terminal output
    coverage html    # generate a HTML form report in htmlcov/index.html


Interface system (legacy)
-------------------------
Print all interfaces and units

.. code-block:: python

    ti.core.print_all_units()

Serialization (legacy)
----------------------

The serialization module of taichi allows you to serialize/deserialize objects into/from binary strings.

You can use ``TI_IO`` macros to explicitly define fields necessary in Taichi.

.. code-block:: cpp

    // TI_IO_DEF
    struct Particle {
        Vector3f position, velocity;
        real mass;
        string name;

        TI_IO_DEF(position, velocity, mass, name);
    }

    // TI_IO_DECL
    struct Particle {
        Vector3f position, velocity;
        real mass;
        bool has_name
        string name;

        TI_IO_DECL() {
            TI_IO(position);
            TI_IO(velocity);
            TI_IO(mass);
            TI_IO(has_name);
            // More flexibility:
            if (has_name) {
                TI_IO(name);
            }
        }
    }

    // TI_IO_DEF_VIRT();


Progress notification (legacy)
------------------------------

The Taichi messenger can send an email to ``$TI_MONITOR_EMAIL`` when the task finishes or crashes.
To enable:

.. code-block:: python

    from taichi.tools import messenger
    messenger.enable(task_id='test')
