---
sidebar_position: 8
---

# Export Taichi kernels to C source

The C backend of Taichi allows you to **export Taichi kernels to C
source**.

The exported Taichi program consists purely of C99-compatible code and
does not require Python. This allows you to use the exported code in a
C/C++ project, or even to further compile it to Javascript/Web Assembly
via Emscripten.

Each C function corresponds to one Taichi kernel. For example,
`Tk_init_c6_0()` may correspond to `init()` in `mpm88.py`.

The exported C code is self-contained for portability. Required Taichi
runtime functions are included in the code.

For example, this allows programmers to distribute Taichi programs in a
binary format, by compiling and linking exported C code to their
project.

:::caution
Currently, this feature is only officially supported on the C backend on
Linux. In the future, we will support macOS and Windows.
:::

## The workflow of exporting

Use `ti.core.start_recording` in the Taichi program you want to export.

Suppose you want to export
[examples/mpm88.py](https://github.com/taichi-dev/taichi/blob/master/examples/mpm88.py),
here is the workflow:

### Export YAML

First, modify `mpm88.py` as shown below:

```python
import taichi as ti

ti.aot.start_recording('mpm88.yml')
ti.init(arch=ti.cc)

... # your program
```

Then please execute `mpm88.py`. Close the GUI window once particles are
shown up correctly.

This will save all the kernels in `mpm88.py` to `mpm88.yml`:

```yaml
- action: "compile_kernel"
   kernel_name: "init_c6_0"
   kernel_source: "void Tk_init_c6_0(struct Ti_Context *ti_ctx) {\n  for (Ti_i32 tmp0 = 0; tmp0 < 8192...\n"
 - action: "launch_kernel"
   kernel_name: "init_c6_0"
 ...
```

:::note

Equivalently, you may also specify these two arguments from environment
variables on Unix-like system:

```bash
TI_ARCH=cc TI_ACTION_RECORD=mpm88.yml python mpm88.py
```

:::

### Compose YAML into a single C file

Now, all necessary information is saved in `mpm88.yml`, in the form of
multiple separate records. You may want to **compose** the separate
kernels into **one single file** for more portability.

We provide a useful CLI tool to do this:

```bash
python3 -m taichi cc_compose mpm88.yml mpm88.c mpm88.h
```

This composes all the kernels and runtimes in `mpm88.yml` into a single
C source file `mpm88.c`:

```c
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
```

... and a C header file `mpm88.h` for declarations of data structures,
functions (Taichi kernels) for this file.

:::note

The generated C source is promised to be C99 compatible.

It should also be functional when compiled using a C++ compiler.
:::

## Calling the exported kernels

Then, link the C file (`mpm88.c`) against your C/C++ project. Include
the header file (`mpm88.h`) when Taichi kernels are called.

For example, calling kernel `init_c6_0` can be implemented as follows:

```cpp
#include "mpm88.h"

int main(void) {
    ...
    Tk_init_c6_0(&Ti_ctx);
    ...
}
```

Alternatively, if you need multiple Taichi contexts within one program:

```cpp
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
```

### Specifying scalar arguments

To specify scalar arguments for kernels:

```cpp
Ti_ctx.args[0].val_f64 = 3.14;  // first argument, float64
Ti_ctx.args[1].val_i32 = 233;  // second argument, int32
Tk_my_kernel_c8_0(&Ti_ctx);
double ret = Ti_ctx.args[0].val_f64;  // return value, float64

printf("my_kernel(3.14, 233) = %lf\n", ret);
```

### Passing external arrays

To pass external arrays as arguments for kernels:

```cpp
float img[640 * 480 * 3];

Ti_ctx.args[0].ptr_f32 = img;  // first argument, float32 pointer to array

// specify the shape of that array:
Ti_ctx.earg[0 * 8 + 0] = 640;  // img.shape[0]
Ti_ctx.earg[0 * 8 + 1] = 480;  // img.shape[1]
Ti_ctx.earg[0 * 8 + 2] = 3;    // img.shape[2]
Tk_matrix_to_ext_arr_c12_0(&Ti_ctx);

// note that the array used in Taichi is row-major:
printf("img[3, 2, 1] = %f\n", img[(3 * 480 + 2) * 3 + 1]);
```

## Taichi.js (WIP)

Once you have C source file generated, you can compile them into
Javascript or WASM via Emscripten.

We provide [Taichi.js](https://github.com/taichi-dev/taichi.js) as an
infrastructure for wrapping Taichi kernels for Javascript. See [its
README.md](https://github.com/taichi-dev/taichi.js/blob/master/README.md)
for the complete workflow.

Check out [this page](https://taichi-dev.github.io/taichi.js) for online
demos.

## Calling Taichi kernels from Julia (WIP)

Once you have C source generated, you can then compile the C source into
a shared object. Then it can be called from other langurages that
provides a C interface, including but not limited to Julia, Matlab,
Mathematica, Java, etc.

TODO: WIP.
