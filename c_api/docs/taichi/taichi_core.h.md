---
sidebar_position: 1
---

# Core Functionality

Taichi Core exposes all necessary interfaces for offloading the AOT modules to Taichi. The following are a list of features that are available regardless of your backend. The corresponding APIs are still under development and subject to change.

## Availability

Taichi C-API has bridged the following backends:

|Backend|Offload Target|Maintenance Tier|
|-|-|-|
|Vulkan|GPU|Tier 1|
|CUDA (LLVM)|GPU (NVIDIA)|Tier 1|
|CPU (LLVM)|CPU|Tier 1|
|DirectX 11|GPU (Windows)|N/A|
|Metal|GPU (macOS, iOS)|N/A|
|OpenGL|GPU|N/A|

The backends with tier 1 support are the most intensively developed and tested ones. In contrast, you would expect a delay in fixes against minor issues on tier 2 backends. The backends currently unsupported might become supported. Among all the tier 1 backends, Vulkan has the most outstanding cross-platform compatibility, so most of the new features will be first available on Vulkan.

For convenience, in the following text (and other C-API documentations), the term **host** refers to the user of the C-API; the term **device** refers to the logical (conceptual) compute device that Taichi Runtime offloads its compute tasks to. A *device* might not be an actual discrete processor away from the CPU and the *host* MAY NOT be able to access the memory allocated on the *device*.

Unless explicitly explained, **device**, **backend**, **offload targer** and **GPU** are used interchangeably; **host**, **user code**, **user procedure** and **CPU** are used interchangeably too.

## How to...

In this section we give an brief introduction about what you might want to do with the Taichi C-API.

### Create and destroy a Runtime Instance

To work with Taichi, you first create an runtime instance. You SHOULD only create a single runtime per thread. Currently we don't officially claim that multiple runtime instances can coexist in a process, please feel free to [report issues](https://github.com/taichi-dev/taichi/issues) if you encountered any problem with such usage.

```cpp
TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
```

When your program reaches the end, you SHOULD destroy the runtime instance. Please ensure any other related resources have been destroyed before the `handle.runtime` itself.

```cpp
ti_destroy_runtime(runtime);
```

### Allocate and Free Memory

Allocate a piece of memory that is visible only to the device. On the GPU backends, it usually means that the memory is located in the graphics memory (GRAM).

```cpp
TiMemoryAllocateInfo mai {};
mai.size = 1024; // Size in bytes.
mai.usage = TI_MEMORY_USAGE_STORAGE_BIT;
TiMemory memory = ti_allocate_memory(runtime, &mai);
```

You MAY free allocated memory explicitly; but memory allocations will be automatically freed when the related `handle.runtime` is destroyed.

```cpp
ti_free_memory(runtime, memory);
```

### Allocate Host-Accessible Memory

By default, memory allocations are physically or conceptually local to the offload target for performance reasons. You can configure the allocate info to enable host access to memory allocations. But please note that host-accessible allocations MAY slow down computation on GPU because of the limited bus bandwidth between the host memory and the device.

You *must* set `host_write` to `true` to allow streaming data to the memory.

```cpp
TiMemoryAllocateInfo mai {};
mai.size = 1024; // Size in bytes.
mai.host_write = true;
mai.usage = TI_MEMORY_USAGE_STORAGE_BIT;
TiMemory steaming_memory = ti_allocate_memory(runtime, &mai);

// ...

std::vector<uint8_t> src = some_random_data_source();

void* dst = ti_map_memory(runtime, steaming_memory);
std::memcpy(dst, src.data(), src.size());
ti_unmap_memory(runtime, streaming_memory);
```

To read data back to the host, `host_read` MUST be set true.

```cpp
TiMemoryAllocateInfo mai {};
mai.size = 1024; // Size in bytes.
mai.host_write = true;
mai.usage = TI_MEMORY_USAGE_STORAGE_BIT;
TiMemory read_back_memory = ti_allocate_memory(runtime, &mai);

// ...

std::vector<uint8_t> dst(1024);
void* src = ti_map_memory(runtime, read_back_memory);
std::memcpy(dst.data(), src, dst.size());
ti_unmap_memory(runtime, read_back_memory);

ti_free_memory(runtime, read_back_memory);
```

**NOTE** `host_read` and `host_write` can be set true simultaneously.

### Load and destroy a Taichi AOT Module

You can load a Taichi AOT module from the filesystem.

```cpp
TiAotModule aot_module = ti_load_aot_module(runtime, "/path/to/aot/module");
```

`/path/to/aot/module` should point to the directory that contains a `metadata.tcb`.

You can destroy an unused AOT module if you have done with it; but please ensure there is no kernel or compute graph related to it pending to `function.submit`.

```cpp
ti_destroy_aot_module(aot_module);
```

### Launch Kernels and Compute Graphs

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
arg1.type = TI_ARGUMENT_TYPE_NDARRAY;
arg1.value.ndarray = ndarray;

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

When you have launched all kernels and compute graphs for this batch, you should `function.submit` and `function.wait` for the execution to finish.

```cpp
ti_submit(runtime);
ti_wait(runtime);
```

**WARNING** This part is subject to change. We're gonna introduce multi-queue in the future.

## API Reference

`alias.bool`

A boolean value. Can be either `definition.true` or `definition.false`. Assignment with other values could lead to undefined behavior.

`definition.true`

A condition or a predicate is satisfied; a statement is valid.

`definition.false`

A condition or a predicate is not satisfied; a statement is invalid.

`alias.flags`

A bit field that can be used to represent 32 orthogonal flags. Bits unspecified in the corresponding flag enum are ignored.

**NOTE** Enumerations and bit-field flags in the C-API have a `TI_XXX_MAX_ENUM` case to ensure the enum to have a 32-bit range and in-memory size. It has no semantical impact and can be safely ignored.

`definition.null_handle`

A sentinal invalid handle that will never be produced from a valid call to Taichi C-API.

`handle.runtime`

Taichi runtime represents an instance of a logical backend and its internal dynamic state. The user is responsible to synchronize any use of `handle.runtime`. The user MUST NOT manipulate multiple `handle.runtime`s in a same thread.

`handle.aot_module`

An ahead-of-time (AOT) compiled Taichi module, which contains a collection of kernels and compute graphs.

`handle.event`

A synchronization primitive to manage on-device execution flows in multiple queues.

`handle.memory`

A contiguous allocation of on-device memory.

`handle.kernel`

A Taichi kernel that can be launched on device for execution.

`handle.compute_graph`

A collection of Taichi kernels (a compute graph) to be launched on device in predefined order.

`enumeration.arch`

Types of backend archs.

- `enumeration.arch.x64`: x64 native CPU backend.
- `enumeration.arch.arm64`: Arm64 native CPU backend.
- `enumeration.arch.cuda`: NVIDIA CUDA GPU backend.
- `enumeration.arch.vulkan`: Vulkan GPU backend.

`enumeration.data_type`

Elementary (primitive) data types. There might be vendor-specific constraints on the available data types so it's recommended to use 32-bit data types if multi-platform distribution is desired.

- `enumeration.data_type.f16`: 16-bit IEEE 754 floating-point number.
- `enumeration.data_type.f32`: 32-bit IEEE 754 floating-point number.
- `enumeration.data_type.f64`: 64-bit IEEE 754 floating-point number.
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

- `enumeration.argument_type.i32`: Signed 32-bit integer.
- `enumeration.argument_type.f32`: Signed 32-bit floating-point number.
- `enumeration.argument_type.ndarray`: ND-array wrapped around a `handle.memory`.

`bit_field.memory_usage`

Usages of a memory allocation.

- `bit_field.memory_usage.storage`: The memory can be read/write accessed by any shader, you usually only need to set this flag.
- `bit_field.memory_usage.uniform`: The memory can be used as a uniform buffer in graphics pipelines.
- `bit_field.memory_usage.vertex`: The memory can be used as a vertex buffer in graphics pipelines.
- `bit_field.memory_usage.index`: The memory can be used as a index buffer in graphics pipelines.

`structure.memory_allocate_info`

Parameters of a newly allocated memory.

- `structure.memory_allocate_info.size`: Size of the allocation in bytes.
- `structure.memory_allocate_info.host_write`: True if the host needs to write to the allocated memory.
- `structure.memory_allocate_info.host_read`: True if the host needs to read from the allocated memory.
- `structure.memory_allocate_info.export_sharing`: True if the memory allocation needs to be exported to other backends (e.g., from Vulkan to CUDA).
- `structure.memory_allocate_info.usage`: All possible usage of this memory allocation. In most of the cases, `bit_field.memory_usage.storage` is enough.

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

Multi-dimentional array of dense primitive data.

- `structure.nd_array.memory`: Memory bound to the ND-array.
- `structure.nd_array.shape`: Shape of the ND-array.
- `structure.nd_array.elem_shape`: Shape of the ND-array elements. You usually need to set this if it's a vector or matrix ND-array.
- `structure.nd_array.elem_type`: Primitive data type of the ND-array elements.

`union.argument_value`

A scalar or structured argument value.

- `union.argument_value.i32`: Value of a 32-bit one's complement signed integer.
- `union.argument_value.f32`: Value of a 32-bit IEEE 754 floating-poing number.
- `union.argument_value.ndarray`: An ND-array to be bound.

`structure.argument`

An argument value to feed kernels.

- `structure.argument.type`: Type of the argument.
- `structure.argument.value`: Value of the argument.

`structure.named_argument`

An named argument value to feed compute graphcs.

- `structure.named_argument.name`: Name of the argument.
- `structure.named_argument.argument`: Argument body.

`function.create_runtime`

Create a Taichi Runtime with the specified `enumeration.arch`.

`function.destroy_runtime`

Destroy a Taichi Runtime.

`function.allocate_memory`

Allocate a contiguous on-device memory with provided parameters.

`function.free_memory`

Free a memory allocation.

`function.map_memory`

Map an on-device memory to a host-addressible space. The user MUST ensure the device is not being used by any device command before the map.

`function.unmap_memory`

Unmap an on-device memory and make any host-side changes about the memory visible to the device. The user MUST ensure there is no further access to the previously mapped host-addressible space.

`function.create_event`

Creates an event primitive.

`function.destroy_event`

Destroys an event primitive.

`function.copy_memory_device_to_device`

Copy the content of a contiguous subsection of on-device memory to another. The two subsections MUST NOT overlap.

`function.launch_kernel`

Launch a Taichi kernel with provided arguments. The arguments MUST have the same count and types in the same order as in the source code.

`function.launch_compute_graph`

Launches a Taichi compute graph with provided named arguments. The named arguments *must* have the same count, names, and types as in the source code.

`function.signal_event`

Sets an event primitive to a signaled state so that the queues waiting for it can go on execution. If the event has been signaled, you *must* call `function.reset_event` to reset it; otherwise, an undefined behavior would occur.

`function.reset_event`

Sets a signaled event primitive back to an unsignaled state.

`function.wait_event`

Wait on an event primitive until it transitions to a signaled state. The user MUST signal the awaited event; otherwise it is an undefined behavior.

`function.submit`

Submit all commands to the logical device for execution. Ensure that any previous device command has been offloaded to the logical computing device.

`function.wait`

Waits until all previously invoked device commands are executed.

`function.load_aot_module`

Load a precompiled AOT module from the filesystem. `definition.null_handle` is returned if the runtime failed to load the AOT module from the given path.

`function.destroy_aot_module`

Destroys a loaded AOT module and releases all related resources.

`function.get_aot_module_kernel`

Retrieves a pre-compiled Taichi kernel from the AOT module. 
Returns `definition.null_handle` if the module does not have a kernel of the specified name.

`function.get_aot_module_compute_graph`

Get a precompiled compute graph from the AOt module. `definition.null_handle` is returned if the module does not have a kernel of the specified name.
