---
sidebar_position: 1
---

# Run a Taichi Program using Ndarray on Android

Taichi's JIT (Just In Time) module compiles a Taichi kernel to the compute shaders according to the specified backend (`arch` in `ti.init()`) and executes these shaders in Taichi's JIT runtime. Taichi's AOT (Ahead of Time) module, however, builds and saves the necessary compute shaders so that you can load and execute these shaders in your own runtime without a Python environment.

Taking a simulation of celestial bodies' orbits as an example, this tutorial walks you through the process of running a Taichi program using Ndarray on Android.

> [Taichi's AOT (Ahead Of Time) module](https://github.com/taichi-dev/taichi/issues/3642) is currently a proof of concept under development and subject to change in the future.

## A definition of Ndarray

Taichi provides a data container called Ndarray.  An Ndarray is a multidimensional container of elements of the same type and size; an element in an Ndarray is virtually a scalar or a tensor.

### Ndarray shape

Ndarray shape defines the Ndarray's layout; element shape defines the element's layout. For example:

- An Ndarray with an Ndarray shape of [2, 1024] and an element shape of [] is an array of 2 x 1,024 = 2,048 scalars.
- An Ndarray with an Ndarray shape of [128, 128] and an element shape of [2, 4] is an array of  128 x 128 = 16,384 2 x 4 matrices.

### Ndarray dimension

The dimension here refers to the number of dimensions of an Ndarray. For example:

- The dimension of an Ndarray with a shape of [1, 2, 3] is three.
- The dimension of an Ndarray with a shape of [500] is one.

### Benefit of Ndarray

Each Ndarray has a fixed dimension but gives you the flexibility of changing its shape in accordance with its dimension.

Unlike a field's shape, which requires you to rewrite and recompile your Taichi program once it is changed, an Ndarray's shape can be *dynamically* changed without the need to recompile.

Taking the simulation of celestial bodies' orbits as an example, suppose you wish to double the number of your celestial bodies to 2,000:

- With Taichi field, you have to compile twice;
- With Ndarray, all you need to do is to update your runtime program.

## Run a Taichi program using Ndarray on Android

The following section walks you through the process of running a Taichi program using Ndarray on Android.

1. [Generate necessary compute shaders](#generate-necessary-compute-shaders)
2. [Parse the generated JSON file](#parse-the-generated-json-file)
3. [Prepare SSBO and shape information](#prepare-ssbo-and-shape-information)
4. [Prepare rendering shaders](#prepare-rendering-shaders)
5. [Execute all shaders](#execute-all-shaders)

:::note

From Step 2, you are required to come up with your own runtime program. We provide an [example Java runtime program for Android](https://github.com/taichi-dev/taichi-aot-demo/blob/master/nbody_ndarray/java_runtime/NbodyNdarray.java) for your reference, but you may need to adapt these codes for your platform and programming language.

:::

### Generate necessary compute shaders

The following Python script defines a Taichi AOT module for generating and saving the necessary compute shaders (GLES shaders in this case) based on the chosen backend (OpenGL).

> Taichi kernels and compute shaders are *not* one-to-one mapping. Each Taichi kernel can generate multiple compute shaders, the number *usually* comparable to that of the loops in the kernel.



```python
import taichi as ti

ti.init(arch=ti.opengl, use_gles=True, allow_nv_shader_extension=False)

# Define constants for computation
G = 1
PI = 3.141592653
N = 1000
m = 5
galaxy_size = 0.4
planet_radius = 1
init_vel = 120
h = 1e-5
substepping = 10

# Define Taichi kernels
@ti.kernel
def initialize(pos: ti.any_arr(element_dim=1), vel: ti.any_arr(element_dim=1)):
    center=ti.Vector([0.5, 0.5])
    for i in pos:
        theta = ti.random() * 2 * PI
        r = (ti.sqrt(ti.random()) * 0.7 + 0.3) * galaxy_size
        offset = r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        pos[i] = center+offset
        vel[i] = [-offset.y, offset.x]
        vel[i] *= init_vel

@ti.kernel
def compute_force(pos: ti.any_arr(element_dim=1), vel: ti.any_arr(element_dim=1), force: ti.any_arr(element_dim=1)):
    for i in pos:
        force[i] = ti.Vector([0.0, 0.0])
    for i in pos:
        p = pos[i]
        for j in pos:
            if i != j:
                diff = p-pos[j]
                r = diff.norm(1e-5)
                f = -G * m * m * (1.0/r)**3 * diff
                force[i] += f
    dt = h/substepping
    for i in pos:
        vel[i].atomic_add(dt*force[i]/m)
        pos[i].atomic_add(dt*vel[i])

# Define Ndarrays
pos = ti.Vector.ndarray(2, ti.f32, N)
vel = ti.Vector.ndarray(2, ti.f32, N)
force = ti.Vector.ndarray(2, ti.f32, N)

# Run the AOT module builder
def aot():
    m = ti.aot.Module(ti.opengl)
    m.add_kernel(initialize, (pos, vel))
    m.add_kernel(compute_force, (pos, vel, force))

    dir_name = 'nbody_aot'
    m.save(dir_name, '')
aot()
```

**In line 3, you initialize Taichi:**

1. Set `use_gles` to `True` to generate GLES compute shaders for Android.
2. Set `allow_nv_shader_extension` to `False` to prevent the generated GLES compute shaders from using Nvidia GL extensions on Android.

> This setting is because Android supports GLES APIs but GLES does not support `NV_SHADER_EXTENSION`.

**In line 50-58, you define and build the Taichi AOT module:**

1. Create a Taichi AOT module, specifying its backend as OpenGL:

```python
     m = ti.aot.Module(ti.opengl)
```

2. Add the required kernels `initialize` and `compute_force`, each with its own Ndarrays, to the module:

```python
m.add_kernel(initialize, (pos, vel))

m.add_kernel(compute_force, (pos, vel, force))
```

3. Specify a folder under your current working directory for holding the files that the module generates:

```python
dir_name = 'nbody_aot'

m.save(dir_name, '')
```

*The necessary compute shaders together with a JSON file appear under the specified directory.*

### Parse the generated JSON file

:::note

From this section, you are required to come up with your own runtime program. We provide an [example Java runtime program for Android](https://github.com/taichi-dev/taichi-aot-demo/blob/master/nbody_ndarray/java_runtime/NbodyNdarray.java) for your reference, but you may need to adapt these codes for your platform and programming language.

:::

After generating the necessary GLES compute shaders, you need to write your runtime program to parse the following JSON file to some data structures. The JSON file contains all the necessary information for executing the compute shaders. Organized by Taichi kernel, it provides a clearer image of the compute shaders and Ndarrays in each kernel. Let's take a closer look at the structure.

> Here, the JSON object for the `compute_force` kernel is omitted for brevity. For a complete JSON file, see [metadata.json](https://github.com/taichi-dev/taichi-aot-demo/blob/master/nbody_ndarray/res/metadata.json).

- **Organized by Taichi kernel**

  - `initialize` (line 4)
  - `compute_force` (line 51)

- **Kernel-specific compute shaders**

  Taking `initialize` as an example, the kernel has generated one compute shader named `initialize_c54_00` (line 7) and the other named `initialize_c54_01` (line 13).

- **Kernel-specific** `args_buff`

  The `initialize` kernel is assigned an `args_buffer` of `128` Bytes (line 21). Note that the size of `args_buffer` is dependent on the number of Ndarrays (`pos` and `vel`) that the kernel takes, (see `arg_count` in line 19). The `initialize` kernel, or each kernel more precisely, has a dedicated `args_buffer` for storing scalar arguments specified in `scalar_args` (line 27) and Ndarray shape information in accordance with what `array_args` (line 28-45) specifies.

  Ndarrays' shape information is organized by their argument index in the `array_args` JSON array: `0` (line 29) corresponds to the `pos` Ndarray, and `1` (line 37) corresponds to the `vel` Ndarray. The argument index is determined by the sequence by which you pass in the Ndarrays when calling `add_kernel()`. See line 53 in the Python script.

  The `pos` Ndarray's shape information in `args_buffer` has an offset of  `64` Bytes in `args_buffer` (line 64). According to line 35 and line 43, the `pos` Ndarray's shape information occupies 96 - 64 = 32 Bytes in `args_buffer`.

  :::tip ATTENTION
  The JSON file only specifies the dimension of the corresponding Ndarray (line 30, 38), allowing you to dynamically update an Ndarray's shape in your runtime program.
  :::

- **Kernel-specific binding index**

  `used.arr_arg_to_bind_idx` (line 46) maps the SSBO of each Ndarray in the kernel to a "more global" binding index for the compute shaders. For example, `"1": 5` (line 48) binds the `vel` Ndarray to the binding index `5`.

```json
{
  "aot_data": {
    "kernels": {
      "initialize": {
        "tasks": [
          {
            "name": "initialize_c54_00",
            "src": "nbody_aot/initialize_c54_00.glsl",
            "workgroup_size": 1,
            "num_groups": 1
          },
          {
            "name": "initialize_c54_01",
            "src": "nbody_aot/initialize_c54_01.glsl",
            "workgroup_size": 128,
            "num_groups": 256
          }
        ],
        "arg_count": 2,
        "ret_count": 0,
        "args_buf_size": 128,
        "ret_buf_size": 0,
        "ext_arr_access": {
          "0": 2,
          "1": 3
        },
        "scalar_args": {},
        "arr_args": {
          "0": {
            "field_dim": 1,
            "is_scalar": false,
            "element_shape": [
              2
            ],
            "shape_offset_in_bytes_in_args_buf": 64
          },
          "1": {
            "field_dim": 1,
            "is_scalar": false,
            "element_shape": [
              2
            ],
            "shape_offset_in_bytes_in_args_buf": 96
          }
        },
        "used.arr_arg_to_bind_idx": {
          "0": 4,
          "1": 5
        }
      },
      "compute_force": {...}
      }
    },
    "kernel_tmpls": {},
    "fields": [],
    "root_buffer_size": 0
  }
}
```
The following provides a detailed description of the keys in the generated JSON file:

`aot_data`: The overarching JSON object.
 - `kernels`: All Taichi kernels.
	- `$(kernel_name)`: Name of a specific Taichi kernel.
		- `tasks`: A JSON array of the generated compute shaders.
			- `name`: Name of a specific compute shader.
			- `src`: Relative path to the shader file.
			- `workgroup_size`: N/A
			- `num_groups`: N/A
		- `arg_count`: Number of the arguments that the Taichi kernel takes.
		- `ret_count`: Number of the values that the Taichi kernel returns.
		- `args_buf_size`: The size of `args_buf` in Bytes.
		- `ret_buf_size`: The size of `ret_buf` in Bytes.
		- `scalar_args`: Scalar arguments that the kernel takes.
		- `arr_args`: Shape information of the Ndarrays in the kernel.
			- `$(arg_index)`: Argument index of an Ndarray
				- `field_dim`: The dimension of the Ndarray.
				- `is_scalar`: Whether the elements in the Ndarray are scalar.
				- `element_shape`: An `int` array indicating the shape of each element in the Ndarray.
				- `shape_offset_in_bytes_in_args_buf`: The offset of the Ndarray's shape information in `args_buf`.
		- `used.arr_arg_to_bind_idx`: A map specifying the SSBO to bind for a given Ndarray. For example, `"1": 5` (line 48) binds the `vel` Ndarray to the binding index `5`.

*Well, we hope you were not overwhelmed with that much information coming in all at once. In the following section, we will revisit the JSON file, as well as provide tables and graphs that help illustrate some of the concepts and notions listed above.*

### Prepare SSBO and shape information

Before executing the GLES compute shaders in your runtime program, you need to get all your resources ready, including:

- Bind SSBO for the corresponding buffer
- Bind SSBO for each Ndarray
- Fill `args_buffer` with Ndarray shape information

#### Bind SSBO for the corresponding buffer

The following table lists the buffers commonly used in a Taichi program together with their binding indexes:

| **Buffer**    | **Global/kernel-spedific** | **Storing**                                                  | **Binding index** |
| ------------- | -------------------------- | ------------------------------------------------------------ | ----------------- |
| `root_buffer` | Global                     | All fields with fixed offsets and of fixed sizes.            | `0`               |
| `gtmp_buffer` | Global                     | Global temporary data                                        | `1`               |
| `args_buffer` | Kernel-specific            | Arguments passed to the Taichi kernel <ul><li>Scalar arguments</li> <li>Each Ndarray's shape information:  <ul><li>Shape of the Ndarray</li> <li>Element shape</li></ul></li></ul> | `2`               |

1. You *only* need to bind an SSBO for `root_buffer` if your Taichi script uses at least one field. Skip this step if your script does not involve field. 
2. Bind a small SSBO, say an SSBO of 1,024 Bytes, to `1`, the binding index of `gtmp_buffer`.
3. Bind an SSBO of 64 x 5 = 320 Bytes to `2`, the binding index of `args_buffer`.

#### Bind SSBO for each Ndarray

Before running a specific kernel in your runtime program (the `initialize` kernel for example), you must bind SSBO of a proper size for each Ndarray in the kernel in accordance to the value of `used.arr_arg_to_bind_idx`.

The following is a summary of line 29-49 of the above JSON file:

| Ndarray | Taichi kernel | Dimension | Element shape | Argument index | Binding index |
| ------- | ------------- | --------- | ------------- | -------------- | ------------- |
| `pos`   | `initialize`  | `1`       | `[2]`         | `0`            | `4`           |
| `vel`   | `initialize`  | `1`       | `[2]`         | `1`            | `5`           |

If you give each Ndarray a shape [500], and an element shape [2] (meaning that each element is a 2-D vector):

- Each Ndarray has 500 x 2 = 1,000 numbers
- Because the number type is float (as specified in the above Python script), the size of each Ndarray's SSBO is 1,000 x 4 = 4,000 Bytes.

Therefore you need to:

- Bind an SSBO of 4,000 Bytes to the binding index `4` for the `pos` Ndarray.
- Bind an SSBO of 4,000 Bytes to the binding index `5` for the `vel` Ndarray.

#### Fill `args_buffer` with Ndarray shape information

When explaining the JSON file, we mention that each kernel has a dedicated `args_buffer` for storing scalar arguments specified in `scalar_args` and Ndarray shape information in accordance with what `array_args` specifies. `array_args` does not specify the Ndarray shape, therefore the final step in your preparation is to fill `args_buffer` with  each Ndarray's shape information in your runtime program.

The typical size of an `args_buffer` is 64 + 64 x 4  Bytes. The first 64 Bytes are reserved for scalar arguments; the remaining buffer is then 64 x 4 Bytes. Each Ndarray is allocated 8 x 4 Bytes for storing its shape information (each has *at most* 8 numbers to indicate its shape information), therefore the remaining buffer can store up to 8 Ndarrays' shape information.

- If your Ndarray shape is [100, 200] and element dimension [3, 2], then you fill 100, 200, 3, and 2 in the corresponding location.
- In this case, both `pos` and `vel` have an Ndarray shape of [500] and an element dimension of [2]. Therefore, you fill 500 and 2 in the corresponding locations.

### Prepare rendering shaders

To perform the rendering (drawing celestial bodies in this case), you are required to write a vertex shader and a fragment shader.

### Execute all shaders

When executing shaders in your runtime program, ensure that you bind SSBOs before executing a Taichi kernel and unbind them when you are done.

 Our [example Android Java runtime](https://github.com/taichi-dev/taichi-aot-demo/blob/master/nbody_ndarray/java_runtime/NbodyNdarray.java) does the following:

1. Run the GLES compute shaders in `initialize` once.
2. For each frame:
   1. Run the GLES compute shaders in `compute_force` 10 times.
   2. Run the vertex and fragment shaders once to do the rendering.



## OpenGL-specific Terms & Definitions

### OpenGL ES (GLES)

OpenGL ES (GLES) is the OpenGL APIs for Embedded Systems. According to [its specifications](https://www.khronos.org/api/opengles), a desktop OpenGL driver supports all GLES APIs.

### OpenGL Shading Language (GLSL)

The OpenGL Shading Language (GLSL) is the primary shading language for OpenGL. GLSL is a C-style language supported directly by OpenGL without extensions.

### Shader

A shader is a user-defined program designed for computing or rendering at a certain stage of a graphics processor.

### SSBO (Shader Storage Buffer Object)

Each Taichi kernel can generate multiple compute shaders, which use SSBO (Shader Storage Buffer Object) as buffer for accessing data.

There are two types of SSBOs: One type corresponds to the buffers maintained by Taichi and includes `root_buffer`, `gtmp_buffer`, and `args_buffer`; the other type corresponds to the Ndarrays maintained by developers and used for sharing data.

> You are required to bind the generated shaders to the corresponding SSBOs in your runtime program. The binding index of an Ndarray's SSBO starts off with `4`.
