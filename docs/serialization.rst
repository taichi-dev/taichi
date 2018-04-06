Serialization
===============================================

Definition Macros
==================================

The serialization module of taichi allows you to serialize/deserialize objects into/from binary strings.

You can use TC_IO macros to explicit define fields necessary in Taichi.

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
