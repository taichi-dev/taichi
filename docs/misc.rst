Misc Functions
===============================================

Logging
-------------------------------

.. code-block:: python

    '''
    level can be {}
        tc.TRACE
        tc.DEBUG
        tc.INFO
        tc.WARN
        tc.ERR
        tc.CRITICAL
    '''
    tc.set_logging_level(level)



Trigger GDB when the program crashes:

Python:

.. code-block:: python

    set_gdb_trigger(on=True)


C++:

.. code-block:: C++

    CoreState::set_trigger_gdb_when_crash(true);
