---
sidebar_position: 1
---

# Core Functionality

Taichi Core exposes all necessary interfaces for offloading the AOT modules to Taichi. The following is a list of features that are available regardless of your backend. The corresponding APIs are still under development and subject to change.

## Availability

Taichi C-API intends to support the following backends:

|Backend     |Offload Target   |Maintenance Tier | Stabilized? |
|------------|-----------------|-----------------|-------------|
|Vulkan      |GPU              |Tier 1           | Yes         |
|Metal       |GPU (macOS, iOS) |Tier 2           | No          |
|CUDA (LLVM) |GPU (NVIDIA)     |Tier 2           | No          |
|CPU (LLVM)  |CPU              |Tier 2           | No          |
|OpenGL      |GPU              |Tier 2           | No          |
|OpenGL ES   |GPU              |Tier 2           | No          |
|DirectX 11  |GPU (Windows)    |N/A              | No          |

The backends with tier-1 support are being developed and tested more intensively. And most new features will be available on Vulkan first because it has the most outstanding cross-platform compatibility among all the tier-1 backends.
For the backends with tier-2 support, you should expect a delay in the fixes to minor issues.

For convenience, in the following text and other C-API documents, the term *host* refers to the user of the C-API; the term *device* refers to the logical (conceptual) compute device, to which Taichi's runtime offloads its compute tasks. A *device* may not be a physical discrete processor other than the CPU and the *host* may *not* be able to access the memory allocated on the *device*.

Unless otherwise specified, **device**, **backend**, **offload target**, and **GPU** are interchangeable; **host**, **user code**, **user procedure**, and **CPU** are interchangeable.

## HowTo

The following section provides a brief introduction to the Taichi C-API.

### Create and destroy a Runtime Instance

You *must* create a runtime instance before working with Taichi, and *only* one runtime per thread. Currently, we do not officially claim that multiple runtime instances can coexist in a process, but please feel free to [file an issue with us](https://github.com/taichi-dev/taichi/issues) if you run into any problem with runtime instance coexistence.

```cpp
// Create a Taichi Runtime on Vulkan device at index 0.
TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN, 0);
```

When your program runs to the end, ensure that:
- You destroy the runtime instance,
- All related resources are destroyed before the `handle.runtime` itself.

```cpp
ti_destroy_runtime(runtime);
```

### Allocate and free memory

Allocate a piece of memory that is visible only to the device. On the GPU backends, it usually means that the memory is located in the graphics memory (GRAM).

```cpp
TiMemoryAllocateInfo mai {};
mai.size = 1024; // Size in bytes.
mai.usage = TI_MEMORY_USAGE_STORAGE_BIT;
TiMemory memory = ti_allocate_memory(runtime, &mai);
```

Allocated memory is automatically freed when the related `handle.runtime` is destroyed. You can also manually free the allocated memory.

```cpp
ti_free_memory(runtime, memory);
```

### Allocate host-accessible memory

By default, memory allocations are physically or conceptually local to the offload target for performance reasons. You can configure the `structure.memory_allocate_info` to enable host access to memory allocations. But please note that host-accessible allocations *may* slow down computation on GPU because of the limited bus bandwidth between the host memory and the device.

You *must* set `host_write` to `definition.true` to allow zero-copy data streaming to the memory.

```cpp
TiMemoryAllocateInfo mai {};
mai.size = 1024; // Size in bytes.
mai.host_write = TI_TRUE;
mai.usage = TI_MEMORY_USAGE_STORAGE_BIT;
TiMemory steaming_memory = ti_allocate_memory(runtime, &mai);

// ...

std::vector<uint8_t> src = some_random_data_source();

void* dst = ti_map_memory(runtime, steaming_memory);
std::memcpy(dst, src.data(), src.size());
ti_unmap_memory(runtime, streaming_memory);
```

To read data back to the host, `host_read` *must* be set to `definition.true`.

```cpp
TiMemoryAllocateInfo mai {};
mai.size = 1024; // Size in bytes.
mai.host_read = TI_TRUE;
mai.usage = TI_MEMORY_USAGE_STORAGE_BIT;
TiMemory read_back_memory = ti_allocate_memory(runtime, &mai);

// ...

std::vector<uint8_t> dst(1024);
void* src = ti_map_memory(runtime, read_back_memory);
std::memcpy(dst.data(), src, dst.size());
ti_unmap_memory(runtime, read_back_memory);

ti_free_memory(runtime, read_back_memory);
```

> You can set `host_read` and `host_write` at the same time.

### Load and destroy a Taichi AOT module

You can load a Taichi AOT module from the filesystem.

```cpp
TiAotModule aot_module = ti_load_aot_module(runtime, "/path/to/aot/module");
```

`/path/to/aot/module` should point to the directory that contains a `metadata.json`.

You can destroy an unused AOT module, but please ensure that there is no kernel or compute graph related to it pending to [`ti_flush`](#function-ti_flush).

```cpp
ti_destroy_aot_module(aot_module);
```

### Launch kernels and compute graphs

You can extract kernels and compute graphs from an AOT module. Kernel and compute graphs are a part of the module, so you don't have to destroy them.

```cpp
TiKernel kernel = ti_get_aot_module_kernel(aot_module, "foo");
TiComputeGraph compute_graph = ti_get_aot_module_compute_graph(aot_module, "bar");
```

You can launch a kernel with positional arguments. Please ensure the types, the sizes and the order matches the source code in Python.

```cpp
TiNdArray ndarray{};
ndarray.memory = get_some_memory();
ndarray.shape.dim_count = 1;
ndarray.shape.dims[0] = 16;
ndarray.elem_shape.dim_count = 2;
ndarray.elem_shape.dims[0] = 4;
ndarray.elem_shape.dims[1] = 4;
ndarray.elem_type = TI_DATA_TYPE_F32;

std::array<TiArgument, 3> args{};

TiArgument& arg0 = args[0];
arg0.type = TI_ARGUMENT_TYPE_I32;
arg0.value.i32 = 123;

TiArgument& arg1 = args[1];
arg1.type = TI_ARGUMENT_TYPE_F32;
arg1.value.f32 = 123.0f;

TiArgument& arg2 = args[2];
arg2.type = TI_ARGUMENT_TYPE_NDARRAY;
arg2.value.ndarray = ndarray;

ti_launch_kernel(runtime, kernel, args.size(), args.data());
```

You can launch a compute graph in a similar way. But additionally please ensure the argument names matches those in the Python source.

```cpp
std::array<TiNamedArgument, 3> named_args{};
TiNamedArgument& named_arg0 = named_args[0];
named_arg0.name = "foo";
named_arg0.argument = args[0];
TiNamedArgument& named_arg1 = named_args[1];
named_arg1.name = "bar";
named_arg1.argument = args[1];
TiNamedArgument& named_arg2 = named_args[2];
named_arg2.name = "baz";
named_arg2.argument = args[2];

ti_launch_compute_graph(runtime, compute_graph, named_args.size(), named_args.data());
```

When you have launched all kernels and compute graphs for this batch, you should `function.flush` and `function.wait` for the execution to finish.

```cpp
ti_flush(runtime);
ti_wait(runtime);
```

**WARNING** This part is subject to change. We will introduce multi-queue in the future.

## API Reference

`alias.bool`

A boolean value. Can be either `definition.true` or `definition.false`. Assignment with other values could lead to undefined behavior.

`definition.true`

A condition or a predicate is satisfied; a statement is valid.

`definition.false`

A condition or a predicate is not satisfied; a statement is invalid.

`alias.flags`

A bit field that can be used to represent 32 orthogonal flags. Bits unspecified in the corresponding flag enum are ignored.

> Enumerations and bit-field flags in the C-API have a `TI_XXX_MAX_ENUM` case to ensure the enum has a 32-bit range and in-memory size. It has no semantical impact and can be safely ignored.

`definition.null_handle`

A sentinal invalid handle that will never be produced from a valid call to Taichi C-API.

`handle.runtime`

Taichi runtime represents an instance of a logical backend and its internal dynamic state. The user is responsible to synchronize any use of `handle.runtime`. The user *must not* manipulate multiple `handle.runtime`s in the same thread.

`handle.aot_module`

An ahead-of-time (AOT) compiled Taichi module, which contains a collection of kernels and compute graphs.

`handle.event`

A synchronization primitive to manage device execution flows in multiple queues.

`handle.memory`

A contiguous allocation of device memory.

`handle.image`

A contiguous allocation of device image.

`handle.sampler`

An image sampler. `definition.null_handle` represents a default image sampler provided by the runtime implementation. The filter modes and address modes of default samplers depend on backend implementation.

`handle.kernel`

A Taichi kernel that can be launched on the offload target for execution.

`handle.compute_graph`

A collection of Taichi kernels (a compute graph) to launch on the offload target in a predefined order.

`enumeration.error`

Errors reported by the Taichi C-API.

- `enumeration.error.success`: The Taichi C-API invocation finished gracefully.
- `enumeration.error.not_supported`: The invoked API, or the combination of parameters is not supported by the Taichi C-API.
- `enumeration.error.corrupted_data`: Provided data is corrupted.
- `enumeration.error.name_not_found`: Provided name does not refer to any existing item.
- `enumeration.error.invalid_argument`: One or more function arguments violate constraints specified in C-API documents, or kernel arguments mismatch the kernel argument list defined in the AOT module.
- `enumeration.error.argument_null`: One or more by-reference (pointer) function arguments point to null.
- `enumeration.error.argument_out_of_range`: One or more function arguments are out of its acceptable range; or enumeration arguments have undefined value.
- `enumeration.error.argument_not_found`: One or more kernel arguments are missing.
- `enumeration.error.invalid_interop`: The intended interoperation is not possible on the current arch. For example, attempts to export a Vulkan object from a CUDA runtime are not allowed.
- `enumeration.error.invalid_state`: The Taichi C-API enters an unrecoverable invalid state. Related Taichi objects are potentially corrupted. The users *should* release the contaminated resources for stability. Please feel free to file an issue if you encountered this error in a normal routine.
- `enumeration.error.incompatible_module`: The AOT module is not compatible with the current runtime.

`enumeration.arch`

Types of backend archs.

- `enumeration.arch.vulkan`: Vulkan GPU backend.
- `enumeration.arch.metal`: Metal GPU backend.
- `enumeration.arch.cuda`: NVIDIA CUDA GPU backend.
- `enumeration.arch.x64`: x64 native CPU backend.
- `enumeration.arch.arm64`: Arm64 native CPU backend.
- `enumeration.arch.opengl`: OpenGL GPU backend.
- `enumeration.arch.gles`: OpenGL ES GPU backend.

`enumeration.capability`

Device capabilities.

`structure.capability_level_info`

An integral device capability level. It currently is not guaranteed that a higher level value is compatible with a lower level value.

`enumeration.data_type`

Elementary (primitive) data types. There might be vendor-specific constraints on the available data types so it's recommended to use 32-bit data types if multi-platform distribution is desired.

- `enumeration.data_type.f16`: 16-bit IEEE 754 half-precision floating-point number.
- `enumeration.data_type.f32`: 32-bit IEEE 754 single-precision floating-point number.
- `enumeration.data_type.f64`: 64-bit IEEE 754 double-precision floating-point number.
- `enumeration.data_type.i8`: 8-bit one's complement signed integer.
- `enumeration.data_type.i16`: 16-bit one's complement signed integer.
- `enumeration.data_type.i32`: 32-bit one's complement signed integer.
- `enumeration.data_type.i64`: 64-bit one's complement signed integer.
- `enumeration.data_type.u8`: 8-bit unsigned integer.
- `enumeration.data_type.u16`: 16-bit unsigned integer.
- `enumeration.data_type.u32`: 32-bit unsigned integer.
- `enumeration.data_type.u64`: 64-bit unsigned integer.

`enumeration.argument_type`

Types of kernel and compute graph argument.

- `enumeration.argument_type.i32`: 32-bit one's complement signed integer.
- `enumeration.argument_type.f32`: 32-bit IEEE 754 single-precision floating-point number.
- `enumeration.argument_type.ndarray`: ND-array wrapped around a `handle.memory`.
- `enumeration.argument_type.texture`: Texture wrapped around a `handle.image`.
- `enumeration.argument_type.scalar`: Typed scalar.
- `enumeration.argument_type.tensor`: Typed tensor.


`bit_field.memory_usage`

Usages of a memory allocation. Taichi requires kernel argument memories to be allocated with `bit_field.memory_usage.storage`.

- `bit_field.memory_usage.storage`: The memory can be read/write accessed by any kernel.
- `bit_field.memory_usage.uniform`: The memory can be used as a uniform buffer in graphics pipelines.
- `bit_field.memory_usage.vertex`: The memory can be used as a vertex buffer in graphics pipelines.
- `bit_field.memory_usage.index`: The memory can be used as an index buffer in graphics pipelines.

`structure.memory_allocate_info`

Parameters of a newly allocated memory.

- `structure.memory_allocate_info.size`: Size of the allocation in bytes.
- `structure.memory_allocate_info.host_write`: True if the host needs to write to the allocated memory.
- `structure.memory_allocate_info.host_read`: True if the host needs to read from the allocated memory.
- `structure.memory_allocate_info.export_sharing`: True if the memory allocation needs to be exported to other backends (e.g., from Vulkan to CUDA).
- `structure.memory_allocate_info.usage`: All possible usage of this memory allocation. In most cases, `bit_field.memory_usage.storage` is enough.

`structure.memory_slice`

A subsection of a memory allocation. The sum of `structure.memory_slice.offset` and `structure.memory_slice.size` cannot exceed the size of `structure.memory_slice.memory`.

- `structure.memory_slice.memory`: The subsectioned memory allocation.
- `structure.memory_slice.offset`: Offset from the beginning of the allocation.
- `structure.memory_slice.size`: Size of the subsection.

`structure.nd_shape`

Multi-dimensional size of an ND-array. Dimension sizes after `structure.nd_shape.dim_count` are ignored.

- `structure.nd_shape.dim_count`: Number of dimensions.
- `structure.nd_shape.dims`: Dimension sizes.

`structure.nd_array`

Multi-dimensional array of dense primitive data.

- `structure.nd_array.memory`: Memory bound to the ND-array.
- `structure.nd_array.shape`: Shape of the ND-array.
- `structure.nd_array.elem_shape`: Shape of the ND-array elements. It *must not* be empty for vector or matrix ND-arrays.
- `structure.nd_array.elem_type`: Primitive data type of the ND-array elements.

`bit_field.image_usage`

Usages of an image allocation. Taichi requires kernel argument images to be allocated with `bit_field.image_usage.storage` and `bit_field.image_usage.sampled`.

- `bit_field.image_usage.storage`: The image can be read/write accessed by any kernel.
- `bit_field.image_usage.sampled`: The image can be read-only accessed by any kernel.
- `bit_field.image_usage.attachment`: The image can be used as a color or depth-stencil attachment depending on its format.

`enumeration.image_dimension`

Dimensions of an image allocation.

- `enumeration.image_dimension.1d`: The image is 1-dimensional.
- `enumeration.image_dimension.2d`: The image is 2-dimensional.
- `enumeration.image_dimension.3d`: The image is 3-dimensional.
- `enumeration.image_dimension.1d_array`: The image is 1-dimensional and it has one or more layers.
- `enumeration.image_dimension.2d_array`: The image is 2-dimensional and it has one or more layers.
- `enumeration.image_dimension.cube`: The image is 2-dimensional and it has 6 layers for the faces towards +X, -X, +Y, -Y, +Z, -Z in sequence.

`enumeration.image_layout`

- `enumeration.image_layout.undefined`: Undefined layout. An image in this layout does not contain any semantical information.
- `enumeration.image_layout.shader_read`: Optimal layout for read-only access, including sampling.
- `enumeration.image_layout.shader_write`: Optimal layout for write-only access.
- `enumeration.image_layout.shader_read_write`: Optimal layout for read/write access.
- `enumeration.image_layout.color_attachment`: Optimal layout as a color attachment.
- `enumeration.image_layout.color_attachment_read`: Optimal layout as an input color attachment.
- `enumeration.image_layout.depth_attachment`: Optimal layout as a depth attachment.
- `enumeration.image_layout.depth_attachment_read`: Optimal layout as an input depth attachment.
- `enumeration.image_layout.transfer_dst`: Optimal layout as a data copy destination.
- `enumeration.image_layout.transfer_src`: Optimal layout as a data copy source.
- `enumeration.image_layout.present_src`:  Optimal layout as a presentation source.

`enumeration.format`

Texture formats. The availability of texture formats depends on runtime support.

`structure.image_offset`

Offsets of an image in X, Y, Z, and array layers.

- `structure.image_offset.x`: Image offset in the X direction.
- `structure.image_offset.y`: Image offset in the Y direction. *Must* be 0 if the image has a dimension of `enumeration.image_dimension.1d` or `enumeration.image_dimension.1d_array`.
- `structure.image_offset.z`: Image offset in the Z direction. *Must* be 0 if the image has a dimension of `enumeration.image_dimension.1d`, `enumeration.image_dimension.2d`, `enumeration.image_dimension.1d_array`, `enumeration.image_dimension.2d_array` or `enumeration.image_dimension.cube_array`.
- `structure.image_offset.array_layer_offset`: Image offset in array layers. *Must* be 0 if the image has a dimension of `enumeration.image_dimension.1d`, `enumeration.image_dimension.2d` or `enumeration.image_dimension.3d`.

`structure.image_extent`

Extents of an image in X, Y, Z, and array layers.

- `structure.image_extent.width`: Image extent in the X direction.
- `structure.image_extent.height`: Image extent in the Y direction. *Must* be 1 if the image has a dimension of `enumeration.image_dimension.1d` or `enumeration.image_dimension.1d_array`.
- `structure.image_extent.depth`: Image extent in the Z direction. *Must* be 1 if the image has a dimension of `enumeration.image_dimension.1d`, `enumeration.image_dimension.2d`, `enumeration.image_dimension.1d_array`, `enumeration.image_dimension.2d_array` or `enumeration.image_dimension.cube_array`.
- `structure.image_extent.array_layer_count`: Image extent in array layers. *Must* be 1 if the image has a dimension of `enumeration.image_dimension.1d`, `enumeration.image_dimension.2d` or `enumeration.image_dimension.3d`. *Must* be 6 if the image has a dimension of `enumeration.image_dimension.cube_array`.

`structure.image_allocate_info`

Parameters of a newly allocated image.

- `structure.image_allocate_info.dimension`: Image dimension.
- `structure.image_allocate_info.extent`: Image extent.
- `structure.image_allocate_info.mip_level_count`: Number of mip-levels.
- `structure.image_allocate_info.format`: Image texel format.
- `structure.image_allocate_info.export_sharing`: True if the memory allocation needs to be exported to other backends (e.g., from Vulkan to CUDA).
- `structure.image_allocate_info.usage`: All possible usages of this image allocation. In most cases, `bit_field.image_usage.storage` and `bit_field.image_usage.sampled` enough.

`structure.image_slice`

A subsection of a memory allocation. The sum of `structure.image_slice.offset` and `structure.image_slice.extent` in each dimension cannot exceed the size of `structure.image_slice.image`.

- `structure.image_slice.image`: The subsectioned image allocation.
- `structure.image_slice.offset`: Offset from the beginning of the allocation in each dimension.
- `structure.image_slice.extent`: Size of the subsection in each dimension.
- `structure.image_slice.mip_level`: The subsectioned mip-level.

`structure.texture`

Image data bound to a sampler.

- `structure.texture.image`: Image bound to the texture.
- `structure.texture.sampler`: The bound sampler that controls the sampling behavior of `structure.texture.image`.
- `structure.texture.dimension`: Image Dimension.
- `structure.texture.extent`: Image extent.
- `structure.texture.format`: Image texel format.

`union.scalar_value`

Scalar value represented by a power-of-two number of bits.

**NOTE** The unsigned integer types merely hold the number of bits in memory and doesn't reflect any type of the underlying data. For example, a 32-bit floating-point scalar value is assigned by `*(float*)&scalar_value.x32 = 0.0f`; a 16-bit signed integer is assigned by `*(int16_t)&scalar_vaue.x16 = 1`. The actual type of the scalar is hinted via `structure.scalar.type`.

- `union.scalar_value.x8`: Scalar value that fits into 8 bits.
- `union.scalar_value.x16`: Scalar value that fits into 16 bits.
- `union.scalar_value.x32`: Scalar value that fits into 32 bits.
- `union.scalar_value.x64`: Scalar value that fits into 64 bits.

`structure.scalar`

A typed scalar value.

`union.tensor_value`

Tensor value represented by a power-of-two number of bits.

- `union.tensor_value.x8`: Tensor value that fits into 8 bits.
- `union.tensor_value.x16`: Tensor value that fits into 16 bits.
- `union.tensor_value.x32`: Tensor value that fits into 32 bits.
- `union.tensor_value.x64`: Tensor value that fits into 64 bits.

`structure.tensor_value_with_length`

A tensor value with a length.

`structure.tensor`

A typed tensor value.

`union.argument_value`

A scalar or structured argument value.

- `union.argument_value.i32`: Value of a 32-bit one's complement signed integer. This is equivalent to `union.scalar_value.x32` with `enumeration.data_type.i32`.
- `union.argument_value.f32`: Value of a 32-bit IEEE 754 single-precision floating-poing number. This is equivalent to `union.scalar_value.x32` with `enumeration.data_type.f32`.
- `union.argument_value.ndarray`: An ND-array to be bound.
- `union.argument_value.texture`: A texture to be bound.
- `union.argument_value.scalar`: An scalar to be bound.
- `union.argument_value.tensor`: A tensor to be bound.

`structure.argument`

An argument value to feed kernels.

- `structure.argument.type`: Type of the argument.
- `structure.argument.value`: Value of the argument.

`structure.named_argument`

A named argument value to feed compute graphs.

- `structure.named_argument.name`: Name of the argument.
- `structure.named_argument.argument`: Argument body.

`function.get_version`

Get the current taichi version. It has the same value as `TI_C_API_VERSION` as defined in `taichi_core.h`.

`function.get_available_archs`

Gets a list of available archs on the current platform. An arch is only available if:

1. The Runtime library is compiled with its support;
2. The current platform is installed with a capable hardware or an emulation software.

An available arch has at least one device available, i.e., device index 0 is always available. If an arch is not available on the current platform, a call to `function.create_runtime` with that arch is guaranteed failing.

**WARNING** Please also note that the order or returned archs is *undefined*.

`function.get_last_error`

Gets the last error raised by Taichi C-API invocations. Returns the semantical error code.

- `function.get_last_error.message_size`: Size of textual error message in `function.get_last_error.message`
- `function.get_last_error.message`: Text buffer for the textual error message. Ignored when `message_size` is 0.

`function.set_last_error`

Sets the provided error as the last error raised by Taichi C-API invocations. It can be useful in extended validation procedures in Taichi C-API wrappers and helper libraries.

- `function.set_last_error.error`: Semantical error code.
- `function.set_last_error.message`: A null-terminated string of the textual error message or `nullptr` for empty error message.

`function.create_runtime`

Creates a Taichi Runtime with the specified `enumeration.arch`.

- `function.create_runtime.arch`: Arch of Taichi Runtime.
- `function.create_runtime.device_index`: The index of device in `function.create_runtime.arch` to create Taichi Runtime on.

`function.destroy_runtime`

Destroys a Taichi Runtime.

`function.set_runtime_capabilities`

Force override the list of available capabilities in the runtime instance.

`function.get_runtime_capabilities`

Gets all capabilities available on the runtime instance.

- `function.get_runtime_capabilities.capability_count`: The total number of capabilities available.
- `function.get_runtime_capabilities.capabilities`: Returned capabilities.

`function.allocate_memory`

Allocates a contiguous device memory with provided parameters.

`function.free_memory`

Frees a memory allocation.

`function.map_memory`

Maps a device memory to a host-addressable space. You *must* ensure that the device is not being used by any device command before the mapping.

`function.unmap_memory`

Unmaps a device memory and makes any host-side changes about the memory visible to the device. You *must* ensure that there is no further access to the previously mapped host-addressable space.

`function.allocate_image`

Allocates a device image with provided parameters.

`function.free_image`

Frees an image allocation.

`function.copy_memory_device_to_device`

Copies the data in a contiguous subsection of the device memory to another subsection. The two subsections *must not* overlap.

`function.copy_image_device_to_device`

Copies the image data in a contiguous subsection of the device image to another subsection. The two subsections *must not* overlap.

`function.track_image`

Tracks the device image with the provided image layout. Because Taichi tracks image layouts internally, it is *only* useful to inform Taichi that the image is transitioned to a new layout by external procedures.

`function.transition_image`

Transitions the image to the provided image layout. Because Taichi tracks image layouts internally, it is *only* useful to enforce an image layout for external procedures to use.

`function.launch_kernel`

Launches a Taichi kernel with the provided arguments. The arguments *must* have the same count and types in the same order as in the source code.

`function.launch_compute_graph`

Launches a Taichi compute graph with provided named arguments. The named arguments *must* have the same count, names, and types as in the source code.

`function.flush`

Submits all previously invoked device commands to the offload device for execution.

`function.wait`

Waits until all previously invoked device commands are executed. Any invoked command that has not been submitted is submitted first.

`function.load_aot_module`

Loads a pre-compiled AOT module from the file system.
Returns `definition.null_handle` if the runtime fails to load the AOT module from the specified path.

`function.create_aot_module`

Creates a pre-compiled AOT module from TCM data.
Returns `definition.null_handle` if the runtime fails to create the AOT module from TCM data.

`function.destroy_aot_module`

Destroys a loaded AOT module and releases all related resources.

`function.get_aot_module_kernel`

Retrieves a pre-compiled Taichi kernel from the AOT module.
Returns `definition.null_handle` if the module does not have a kernel of the specified name.

`function.get_aot_module_compute_graph`

Retrieves a pre-compiled compute graph from the AOT module.
Returns `definition.null_handle` if the module does not have a compute graph of the specified name.
