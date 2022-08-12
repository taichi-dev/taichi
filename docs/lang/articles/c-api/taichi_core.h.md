# Taichi C-API: Core Functionality

Taichi Core exposes all necessary interfaces to offload AOT modules to Taichi. Here lists the features universally available disregards to any specific backend. The Taichi Core APIs are guaranteed to be forward compatible.

TODO: (@PENGUINLIONG) Example usage.

## Declarations

---
### Alias `TiBool`

```c
// alias.bool
typedef uint32_t TiBool;
```

A boolean value. Can be either `definition.true` or `definition.false`. Assignment with other values could lead to undefined behavior.

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

Taichi runtime represents an instance of a logical computating device and its internal dynamic states. The user is responsible to synchronize any use of `handle.runtime`.

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

---
### Bit Field `TiMemoryUsageFlagBits`

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

Create a Taichi Runtime with the specified `enumeration.arch`.

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

Set an event primitive to a signaled state, so the queues waiting upon the event can go on execution. If the event has been signaled before, the event MUST be reset with `function.reset_event`; otherwise it is an undefined behavior.

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

Load a precompiled AOT module from the filesystem. `definition.null_handle` is returned if the runtime failed to load the AOT module from the given path.

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

Get a precompiled Taichi kernel from the AOT module. `definition.null_handle` is returned if the module does not have a kernel of the specified name.

---
### Function `ti_get_aot_module_compute_graph`

```c
// function.get_aot_module_compute_graph
TI_DLL_EXPORT TiComputeGraph TI_API_CALL ti_get_aot_module_compute_graph(
  TiAotModule aot_module,
  const char* name
);
```

Get a precompiled compute graph from the AOt module. `definition.null_handle` is returned if the module does not have a kernel of the specified name.
