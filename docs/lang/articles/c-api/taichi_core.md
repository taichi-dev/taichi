---
sidebar_position: 1
---

# Core Functionalities

Taichi Core exposes all necessary interfaces to offload AOT modules to Taichi. Here lists the features universally available disregards to any specific backend. These APIs are still in active development so is subject to change.

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

When your program reaches the end, you SHOULD destroy the runtime instance. Please ensure any other related resources have been destroyed before the `TiRuntime` itself.

```cpp
ti_destroy_runtime(runtime);
```

### Allocate and Free Device-Only Memory

Allocate a piece of memory that is only visible to the device. On GPU backends, it usually means that the memory is located in the graphics memory (GRAM).

```cpp
TiMemoryAllocateInfo mai {};
mai.size = 1024; // Size in bytes.
mai.usage = TI_MEMORY_USAGE_STORAGE_BIT;
TiMemory memory = ti_allocate_memory(runtime, &mai);
```

You MAY free allocated memory explicitly; but memory allocations will be automatically freed when the related `TiRuntime` is destroyed.

```cpp
ti_free_memory(runtime, memory);
```

### Allocate Host-Accessible Memory

To allow data to be streamed into the memory, `host_write` MUST be set true.

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

**NOTE** `host_read` and `host_write` can be set true simultaneously. But please note that host-accessible allocations MAY slow down computation on a GPU because the limited bus bandwidth between the host memory and the device.

### Load and destroy a Taichi AOT Module

You can load a Taichi AOT module from the filesystem.

```cpp
TiAotModule aot_module = ti_load_aot_module(runtime, "/path/to/aot/module");
```

`/path/to/aot/module` should point to the directory that contains a `metadata.tcb`.

You can destroy an unused AOT module if you have done with it; but please ensure there is no kernel or compute graph related to it pending to `ti_submit`.

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

When you have launched all kernels and compute graphs for this batch, you should `ti_submit` and `ti_wait` for the execution to finish.

```cpp
ti_submit(runtime);
ti_wait(runtime);
```

**WARNING** This part is subject to change. We're gonna introduce multi-queue in the future.

## API Reference

### Alias `TiBool`

```c
// alias.bool
typedef uint32_t TiBool;
```

A boolean value. Can be either `TI_TRUE` or `TI_FALSE`. Assignment with other values could lead to undefined behavior.

---
### Definition `TI_FALSE`

```c
// definition.false
#define TI_FALSE 0
```

A condition or a predicate is not satisfied; a statement is invalid.

---
### Definition `TI_TRUE`

```c
// definition.true
#define TI_TRUE 1
```

A condition or a predicate is satisfied; a statement is valid.

---
### Alias `TiFlags`

```c
// alias.flags
typedef uint32_t TiFlags;
```

A bit field that can be used to represent 32 orthogonal flags.

---
### Definition `TI_NULL_HANDLE`

```c
// definition.null_handle
#define TI_NULL_HANDLE 0
```

A sentinal invalid handle that will never be produced from a valid call to Taichi C-API.

---
### Handle `TiRuntime`

```c
// handle.runtime
typedef struct TiRuntime_t* TiRuntime;
```

Taichi runtime represents an instance of a logical computating device and its internal dynamic states. The user is responsible to synchronize any use of `TiRuntime`. The user MUST NOT manipulate multiple `TiRuntime`s in a same thread.

---
### Handle `TiAotModule`

```c
// handle.aot_module
typedef struct TiAotModule_t* TiAotModule;
```

An ahead-of-time (AOT) compiled Taichi module, which contains a collection of kernels and compute graphs.

---
### Handle `TiEvent`

```c
// handle.event
typedef struct TiEvent_t* TiEvent;
```

A synchronization primitive to manage on-device execution flows in multiple queues.

---
### Handle `TiMemory`

```c
// handle.memory
typedef struct TiMemory_t* TiMemory;
```

A contiguous allocation of on-device memory.

---
### Handle `TiKernel`

```c
// handle.kernel
typedef struct TiKernel_t* TiKernel;
```

A Taichi kernel that can be launched on device for execution.

---
### Handle `TiComputeGraph`

```c
// handle.compute_graph
typedef struct TiComputeGraph_t* TiComputeGraph;
```

A collection of Taichi kernels (a compute graph) to be launched on device with predefined order.

---
### Enumeration `TiArch`

```c
// enumeration.arch
typedef enum TiArch {
  TI_ARCH_X64 = 0,
  TI_ARCH_ARM64 = 1,
  TI_ARCH_JS = 2,
  TI_ARCH_CC = 3,
  TI_ARCH_WASM = 4,
  TI_ARCH_CUDA = 5,
  TI_ARCH_METAL = 6,
  TI_ARCH_OPENGL = 7,
  TI_ARCH_DX11 = 8,
  TI_ARCH_OPENCL = 9,
  TI_ARCH_AMDGPU = 10,
  TI_ARCH_VULKAN = 11,
  TI_ARCH_MAX_ENUM = 0xffffffff,
} TiArch;
```

Types of logical offload devices.

---
### Enumeration `TiDataType`

```c
// enumeration.data_type
typedef enum TiDataType {
  TI_DATA_TYPE_F16 = 0,
  TI_DATA_TYPE_F32 = 1,
  TI_DATA_TYPE_F64 = 2,
  TI_DATA_TYPE_I8 = 3,
  TI_DATA_TYPE_I16 = 4,
  TI_DATA_TYPE_I32 = 5,
  TI_DATA_TYPE_I64 = 6,
  TI_DATA_TYPE_U1 = 7,
  TI_DATA_TYPE_U8 = 8,
  TI_DATA_TYPE_U16 = 9,
  TI_DATA_TYPE_U32 = 10,
  TI_DATA_TYPE_U64 = 11,
  TI_DATA_TYPE_GEN = 12,
  TI_DATA_TYPE_UNKNOWN = 13,
  TI_DATA_TYPE_MAX_ENUM = 0xffffffff,
} TiDataType;
```

Elementary (primitive) data types.

---
### Enumeration `TiArgumentType`

```c
// enumeration.argument_type
typedef enum TiArgumentType {
  TI_ARGUMENT_TYPE_I32 = 0,
  TI_ARGUMENT_TYPE_F32 = 1,
  TI_ARGUMENT_TYPE_NDARRAY = 2,
  TI_ARGUMENT_TYPE_MAX_ENUM = 0xffffffff,
} TiArgumentType;
```

Types of kernel and compute graph argument.

- `TI_ARGUMENT_TYPE_I32`: Signed 32-bit integer.
- `TI_ARGUMENT_TYPE_F32`: Signed 32-bit floating-point number.
- `TI_ARGUMENT_TYPE_NDARRAY`: ND-array wrapped around a `TiMemory`.

---
### BitField `TiMemoryUsageFlagBits`

```c
// bit_field.memory_usage
typedef enum TiMemoryUsageFlagBits {
  TI_MEMORY_USAGE_STORAGE_BIT = 1 << 0,
  TI_MEMORY_USAGE_UNIFORM_BIT = 1 << 1,
  TI_MEMORY_USAGE_VERTEX_BIT = 1 << 2,
  TI_MEMORY_USAGE_INDEX_BIT = 1 << 3,
} TiMemoryUsageFlagBits;
typedef TiFlags TiMemoryUsageFlags;
```

Usages of a memory allocation.

- `TI_MEMORY_USAGE_STORAGE_BIT`: The memory can be read/write accessed by any shader, you usually only need to set this flag.
- `TI_MEMORY_USAGE_UNIFORM_BIT`: The memory can be used as a uniform buffer in graphics pipelines.
- `TI_MEMORY_USAGE_VERTEX_BIT`: The memory can be used as a vertex buffer in graphics pipelines.
- `TI_MEMORY_USAGE_INDEX_BIT`: The memory can be used as a index buffer in graphics pipelines.

---
### Structure `TiMemoryAllocateInfo`

```c
// structure.memory_allocate_info
typedef struct TiMemoryAllocateInfo {
  uint64_t size;
  TiBool host_write;
  TiBool host_read;
  TiBool export_sharing;
  TiMemoryUsageFlagBits usage;
} TiMemoryAllocateInfo;
```

Parameters of a newly allocated memory.

- `TiMemoryAllocateInfo.size`: Size of the allocation in bytes.
- `TiMemoryAllocateInfo.host_write`: True if the host needs to write to the allocated memory.
- `TiMemoryAllocateInfo.host_read`: True if the host needs to read from the allocated memory.
- `TiMemoryAllocateInfo.export_sharing`: True if the memory allocation needs to be exported to other backends (e.g., from Vulkan to CUDA).
- `TiMemoryAllocateInfo.usage`: All possible usage of this memory allocation. In most of the cases, `TI_MEMORY_USAGE_STORAGE_BIT` is enough.

---
### Structure `TiMemorySlice`

```c
// structure.memory_slice
typedef struct TiMemorySlice {
  TiMemory memory;
  uint64_t offset;
  uint64_t size;
} TiMemorySlice;
```

A subsection of a memory allocation.

---
### Structure `TiNdShape`

```c
// structure.nd_shape
typedef struct TiNdShape {
  uint32_t dim_count;
  uint32_t dims[16];
} TiNdShape;
```

Multi-dimensional size of an ND-array.

---
### Structure `TiNdArray`

```c
// structure.nd_array
typedef struct TiNdArray {
  TiMemory memory;
  TiNdShape shape;
  TiNdShape elem_shape;
  TiDataType elem_type;
} TiNdArray;
```

Multi-dimentional array of dense primitive data.

---
### Union `TiArgumentValue`

```c
// union.argument_value
typedef union TiArgumentValue {
  int32_t i32;
  float f32;
  TiNdArray ndarray;
} TiArgumentValue;
```

A scalar or structured argument value.

---
### Structure `TiArgument`

```c
// structure.argument
typedef struct TiArgument {
  TiArgumentType type;
  TiArgumentValue value;
} TiArgument;
```

An argument value to feed kernels.

---
### Structure `TiNamedArgument`

```c
// structure.named_argument
typedef struct TiNamedArgument {
  const char* name;
  TiArgument argument;
} TiNamedArgument;
```

An named argument value to feed compute graphcs.

---
### Function `ti_create_runtime`

```c
// function.create_runtime
TI_DLL_EXPORT TiRuntime TI_API_CALL ti_create_runtime(
  TiArch arch
);
```

Create a Taichi Runtime with the specified `TiArch`.

---
### Function `ti_destroy_runtime`

```c
// function.destroy_runtime
TI_DLL_EXPORT void TI_API_CALL ti_destroy_runtime(
  TiRuntime runtime
);
```

Destroy a Taichi Runtime.

---
### Function `ti_allocate_memory`

```c
// function.allocate_memory
TI_DLL_EXPORT TiMemory TI_API_CALL ti_allocate_memory(
  TiRuntime runtime,
  const TiMemoryAllocateInfo* allocate_info
);
```

Allocate a contiguous on-device memory with provided parameters.

---
### Function `ti_free_memory`

```c
// function.free_memory
TI_DLL_EXPORT void TI_API_CALL ti_free_memory(
  TiRuntime runtime,
  TiMemory memory
);
```

Free a memory allocation.

---
### Function `ti_map_memory`

```c
// function.map_memory
TI_DLL_EXPORT void* TI_API_CALL ti_map_memory(
  TiRuntime runtime,
  TiMemory memory
);
```

Map an on-device memory to a host-addressible space. The user MUST ensure the device is not being used by any device command before the map.

---
### Function `ti_unmap_memory`

```c
// function.unmap_memory
TI_DLL_EXPORT void TI_API_CALL ti_unmap_memory(
  TiRuntime runtime,
  TiMemory memory
);
```

Unmap an on-device memory and make any host-side changes about the memory visible to the device. The user MUST ensure there is no further access to the previously mapped host-addressible space.

---
### Function `ti_create_event`

```c
// function.create_event
TI_DLL_EXPORT TiEvent TI_API_CALL ti_create_event(
  TiRuntime runtime
);
```

Create an event primitive.

---
### Function `ti_destroy_event`

```c
// function.destroy_event
TI_DLL_EXPORT void TI_API_CALL ti_destroy_event(
  TiEvent event
);
```

Destroy an event primitive.

---
### Function `ti_copy_memory_device_to_device` (Device Command)

```c
// function.copy_memory_device_to_device
TI_DLL_EXPORT void TI_API_CALL ti_copy_memory_device_to_device(
  TiRuntime runtime,
  const TiMemorySlice* dst_memory,
  const TiMemorySlice* src_memory
);
```

Copy the content of a contiguous subsection of on-device memory to another. The two subsections MUST NOT overlap.

---
### Function `ti_launch_kernel` (Device Command)

```c
// function.launch_kernel
TI_DLL_EXPORT void TI_API_CALL ti_launch_kernel(
  TiRuntime runtime,
  TiKernel kernel,
  uint32_t arg_count,
  const TiArgument* args
);
```

Launch a Taichi kernel with provided arguments. The arguments MUST have the same count and types in the same order as in the source code.

---
### Function `ti_launch_compute_graph` (Device Command)

```c
// function.launch_compute_graph
TI_DLL_EXPORT void TI_API_CALL ti_launch_compute_graph(
  TiRuntime runtime,
  TiComputeGraph compute_graph,
  uint32_t arg_count,
  const TiNamedArgument* args
);
```

Launch a Taichi kernel with provided named arguments. The named arguments MUST have the same count, names and types as in the source code.

---
### Function `ti_signal_event` (Device Command)

```c
// function.signal_event
TI_DLL_EXPORT void TI_API_CALL ti_signal_event(
  TiRuntime runtime,
  TiEvent event
);
```

Set an event primitive to a signaled state, so the queues waiting upon the event can go on execution. If the event has been signaled before, the event MUST be reset with `ti_reset_event`; otherwise it is an undefined behavior.

---
### Function `ti_reset_event` (Device Command)

```c
// function.reset_event
TI_DLL_EXPORT void TI_API_CALL ti_reset_event(
  TiRuntime runtime,
  TiEvent event
);
```

Set a signaled event primitive back to an unsignaled state.

---
### Function `ti_wait_event` (Device Command)

```c
// function.wait_event
TI_DLL_EXPORT void TI_API_CALL ti_wait_event(
  TiRuntime runtime,
  TiEvent event
);
```

Wait on an event primitive until it transitions to a signaled state. The user MUST signal the awaited event; otherwise it is an undefined behavior.

---
### Function `ti_submit`

```c
// function.submit
TI_DLL_EXPORT void TI_API_CALL ti_submit(
  TiRuntime runtime
);
```

Submit all commands to the logical device for execution. Ensure that any previous device command has been offloaded to the logical computing device.

---
### Function `ti_wait`

```c
// function.wait
TI_DLL_EXPORT void TI_API_CALL ti_wait(
  TiRuntime runtime
);
```

Wait until all previously invoked device command has finished execution.

---
### Function `ti_load_aot_module`

```c
// function.load_aot_module
TI_DLL_EXPORT TiAotModule TI_API_CALL ti_load_aot_module(
  TiRuntime runtime,
  const char* module_path
);
```

Load a precompiled AOT module from the filesystem. `TI_NULL_HANDLE` is returned if the runtime failed to load the AOT module from the given path.

---
### Function `ti_destroy_aot_module`

```c
// function.destroy_aot_module
TI_DLL_EXPORT void TI_API_CALL ti_destroy_aot_module(
  TiAotModule aot_module
);
```

Destroy a loaded AOT module and release all related resources.

---
### Function `ti_get_aot_module_kernel`

```c
// function.get_aot_module_kernel
TI_DLL_EXPORT TiKernel TI_API_CALL ti_get_aot_module_kernel(
  TiAotModule aot_module,
  const char* name
);
```

Get a precompiled Taichi kernel from the AOT module. `TI_NULL_HANDLE` is returned if the module does not have a kernel of the specified name.

---
### Function `ti_get_aot_module_compute_graph`

```c
// function.get_aot_module_compute_graph
TI_DLL_EXPORT TiComputeGraph TI_API_CALL ti_get_aot_module_compute_graph(
  TiAotModule aot_module,
  const char* name
);
```

Get a precompiled compute graph from the AOt module. `TI_NULL_HANDLE` is returned if the module does not have a kernel of the specified name.
