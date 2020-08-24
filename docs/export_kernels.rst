Export Taichi kernels for distribution
======================================

The C backend of Taichi allows you to **export Taichi kernels to C source**.

The exported Taichi program, which are simply C99 compatible source,
does not necessary to be launched from Python. Instead, you may use them in
your C/C++ project. Or even calling them in Javascript via Empscripten.

Each C function corresponds to a Taichi kernel.
For example, ``Tk_init_c6_0()`` may corresponds to ``init()`` in ``mpm88.py``.

Only the source is need when copying, all the required Taichi runtimes are
included in that single C source file.

This also allows commercial people to distribute their Taichi program in
binary format by linking this file with their project.

.. note::

    Currently this feature is only officially supported on the C backend.
    Also note that the C backend is only released on **Linux** platform for now.


The workflow for export
-----------------------

Use ``ti.core.start_recording`` in the Taichi program you want to export.

Suppose you want to export the `examples/mpm88.py <https://github.com/taichi-dev/taichi/blob/master/examples/mpm88.py>`_, here are the workflow:

Export YAML
+++++++++++

First, modify the ``mpm88.py`` a little bit:

.. code-block:: python

    import taichi as ti

    ti.core.start_recording('mpm88.yml')
    ti.init(arch=ti.cc)

    ... # your program

And run it. Close the GUI window once particles are shown up correctly.

This will save all the kernels in ``mpm88.py`` to ``mpm88.yml``:

.. code-block:: yaml

   - action: "compile_kernel"
      kernel_name: "init_c6_0"
      kernel_source: "void Tk_init_c6_0(struct Ti_Context *ti_ctx) {\n  for (Ti_i32 tmp0 = 0; tmp0 < 8192...\n"
    - action: "launch_kernel"
      kernel_name: "init_c6_0"
    ...

.. note::

    Equivalently, you may also specify these two arguments from environment
    variables if you are on Unix-alike system:

    .. code-block:: bash

        TI_ARCH=cc TI_ACTION_RECORD=mpm88.yml python mpm88.py

Compose YAML into single C file
+++++++++++++++++++++++++++++++

Now all necessary information are saved in ``mpm88.yml``.
However the ``kernel_source``'s are separated one-by-one.
So you may want to **compose** the separate kernels into **one single file**
to make distribution easier.

We provide a useful tool to compose them, type these commands to your console:

.. code-block:: bash

   python3 -m taichi cc_compose mpm88.yml mpm88.c mpm88.h

This composes all the kernels and runtimes in ``mpm88.yml`` into a single C
source file ``mpm88.c``:

.. code-block:: c

    ...

    Ti_i8 Ti_gtmp[1048576];
    union Ti_BitCast Ti_args[8];
    Ti_i32 Ti_earg[8 * 8];

    struct Ti_Context Ti_ctx = {  // statically-allocated context for convenience!
      &Ti_root, Ti_gtmp, Ti_args, Ti_earg,
    };

    void Tk_init_c6_0(struct Ti_Context *ti_ctx) {
      for (Ti_i32 tmp0 = 0; tmp0 < 8192; tmp0 += 1) {
        Ti_i32 tmp1 = tmp0;
        Ti_f32 tmp2 = Ti_rand_f32();
        Ti_f32 tmp3 = Ti_rand_f32();
        Ti_f32 tmp4 = 0.4;
        Ti_f32 tmp5 = tmp2 * tmp4;

        ...

And a C header file ``mpm88.h`` for declarations of data structures, functions
(kernels) for this file.

.. note::

   The generated C source is promised to be C99 compatible.

   It should be functional when being compiled as C++ as well.


Calling the exported kernels
----------------------------

Then, link this file (``mpm88.c``) together with your C/C++ project.
Include the header (``mpm88.h``) to where kernels are to be called.

To call the kernel ``init_c6_0``, for example:

.. code-block:: cpp

    #include "mpm88.h"

    int main(void) {
        ...
        Tk_init_c6_0(&Ti_ctx);
        ...
    }


Or, if you need multiple Taichi context within one program:

.. code-block:: cpp

    extern "C" {  // if you use mpm88.c instead of renaming it to mpm88.cpp
    #include "mpm88.h"
    }

    class MyRenderer {
      ...
      struct Ti_Context per_renderer_taichi_context;
      ...
    };

    MyRenderer::MyRenderer() {
      // allocate buffers on your own:
      per_renderer_taichi_context.root = malloc(...);
      ...
      Tk_init_c6_0(&per_renderer_taichi_context);
    }


Specifying scalar arguments
+++++++++++++++++++++++++++

To specify scalar arguments for kernels:

.. code-block:: cpp

    Ti_ctx.args[0].val_f64 = 3.14;  // first argument, float64
    Ti_ctx.args[1].val_i32 = 233;  // second argument, int32
    Tk_my_kernel_c8_0(&Ti_ctx);
    double ret = Ti_ctx.args[0].val_f64;  // return value, float64

    printf("my_kernel(3.14, 233) = %lf\n", ret);

Passing external arrays
+++++++++++++++++++++++

To pass external arrays as arguments for kernels:

.. code-block:: cpp

    float img[640 * 480 * 3];

    Ti_ctx.args[0].ptr_f32 = img;  // first argument, float32 pointer to array

    // specify the shape of that array:
    Ti_ctx.earg[0 * 8 + 0] = 640;  // img.shape[0]
    Ti_ctx.earg[0 * 8 + 1] = 480;  // img.shape[1]
    Ti_ctx.earg[0 * 8 + 2] = 3;    // img.shape[2]
    Tk_matrix_to_ext_arr_c12_0(&Ti_ctx);

    // note that the array used in Taichi is row-major:
    printf("img[3, 2, 1] = %f\n", img[(3 * 480 + 2) * 3 + 1]);

Taichi.js (WIP)
---------------

Once you have C source generated, you can compile the C source into Javascript
or WASM via Emscripten.

We provide `Taichi.js <https://github.com/taichi-dev/taichi.js>`_ as an
infrastructure for wrapping Taichi kernels for Javascript.
See `its README.md <https://github.com/taichi-dev/taichi.js/blob/master/README.md>`_ for the complete workflow.

Check `this page <https://taichi-dev.github.io/taichi.js>`_ for online demo.

Calling Taichi from Julia (WIP)
-------------------------------
