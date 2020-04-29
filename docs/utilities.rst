Utilities
==================================

TODO: update

GUI system
----------

.. code-block:: python

    gui = ti.GUI('Title', (640, 480))
    while not gui.get_event(ti.GUI.ESCAPE):  # until ESC is pressed
        gui.set_image(img)
        gui.show()


Also checkout ``examples/keyboard.py`` for more advanced event processing.


Image I/O
---------

.. code-block:: python

    img = ti.imread('hello.png')
    ti.imshow(img, 'Window Title')
    ti.imwrite(img, 'hello2.png')


Logging
-------

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

The default logging level is ``ti.INFO``.
You can also override default logging level by setting the environment variable
``TI_LOG_LEVEL`` to values such as ``trace`` and ``warn``.

Trigger GDB when the program crashes
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

You can use ``TI_IO`` macros to explicit define fields necessary in Taichi.

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


Progress Notification
----------------------------------

The taichi messager can send an email to ``$TI_MONITOR_EMAIL`` when the task finished or crashed.
To enable:

.. code-block:: python

    from taichi.tools import messager
    messager.enable(task_id='test')
