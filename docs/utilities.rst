Utilities
==================================

TODO: update

Logging
----------------------------------

.. code-block:: python

    '''
    level can be {}
        ti.TRACE
        ti.DEBUG
        ti.INFO
        ti.WARN
        ti.ERR
        ti.CRITICAL
    '''
    ti.set_logging_level(level)



Trigger GDB when the program crashes:
--------------------------------------

.. code-block::

  # Python
  ti.set_gdb_trigger(True)

  // C++
  CoreState::set_trigger_gdb_when_crash(true);

Interface System
---------------------------------
Print all interfaces and units

.. code-block:: python

    ti.core.print_all_units()

Serialization
----------------------------------

The serialization module of taichi allows you to serialize/deserialize objects into/from binary strings.

You can use ``TC_IO`` macros to explicit define fields necessary in Taichi.

.. code-block:: cpp

    // TC_IO_DEF
    struct Particle {
        Vector3f position, velocity;
        real mass;
        string name;

        TC_IO_DEF(position, velocity, mass, name);
    }

    // TC_IO_DECL
    struct Particle {
        Vector3f position, velocity;
        real mass;
        bool has_name
        string name;

        TC_IO_DECL() {
            TC_IO(position);
            TC_IO(velocity);
            TC_IO(mass);
            TC_IO(has_name);
            // More flexibility:
            if (has_name) {
                TC_IO(name);
            }
        }
    }

    // TC_IO_DEF_VIRT();


Progress Notification
----------------------------------

The taichi messager can send an email to ``$TC_MONITOR_EMAIL`` when the task finished or crashed.
To enable:

.. code-block:: python

    from taichi.tools import messager
    messager.enable(task_id='test')
