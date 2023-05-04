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
- All related resources are destroyed before the [`TiRuntime`](#handle-tiruntime) itself.

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

Allocated memory is automatically freed when the related [`TiRuntime`](#handle-tiruntime) is destroyed. You can also manually free the allocated memory.

```cpp
ti_free_memory(runtime, memory);
```

### Allocate host-accessible memory

By default, memory allocations are physically or conceptually local to the offload target for performance reasons. You can configure the [`TiMemoryAllocateInfo`](#structure-timemoryallocateinfo) to enable host access to memory allocations. But please note that host-accessible allocations *may* slow down computation on GPU because of the limited bus bandwidth between the host memory and the device.

You *must* set `host_write` to [`TI_TRUE`](#definition-ti_true) to allow zero-copy data streaming to the memory.

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

To read data back to the host, `host_read` *must* be set to [`TI_TRUE`](#definition-ti_true).

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

When you have launched all kernels and compute graphs for this batch, you should [`ti_flush`](#function-ti_flush) and [`ti_wait`](#function-ti_wait) for the execution to finish.

```cpp
ti_flush(runtime);
ti_wait(runtime);
```

**WARNING** This part is subject to change. We will introduce multi-queue in the future.

## API Reference

### Alias `TiBool`

> Stable since Taichi version: 1.4.0

```c
// alias.bool
typedef uint32_t TiBool;
```

A boolean value. Can be either [`TI_TRUE`](#definition-ti_true) or [`TI_FALSE`](#definition-ti_false). Assignment with other values could lead to undefined behavior.

---
### Definition `TI_FALSE`

> Stable since Taichi version: 1.4.0

```c
// definition.false
#define TI_FALSE 0
```

A condition or a predicate is not satisfied; a statement is invalid.

---
### Definition `TI_TRUE`

> Stable since Taichi version: 1.4.0

```c
// definition.true
#define TI_TRUE 1
```

A condition or a predicate is satisfied; a statement is valid.

---
### Alias `TiFlags`

> Stable since Taichi version: 1.4.0

```c
// alias.flags
typedef uint32_t TiFlags;
```

A bit field that can be used to represent 32 orthogonal flags. Bits unspecified in the corresponding flag enum are ignored.

> Enumerations and bit-field flags in the C-API have a `TI_XXX_MAX_ENUM` case to ensure the enum has a 32-bit range and in-memory size. It has no semantical impact and can be safely ignored.

---
### Definition `TI_NULL_HANDLE`

> Stable since Taichi version: 1.4.0

```c
// definition.null_handle
#define TI_NULL_HANDLE 0
```

A sentinal invalid handle that will never be produced from a valid call to Taichi C-API.

---
### Handle `TiRuntime`

> Stable since Taichi version: 1.4.0

```c
// handle.runtime
typedef struct TiRuntime_t* TiRuntime;
```

Taichi runtime represents an instance of a logical backend and its internal dynamic state. The user is responsible to synchronize any use of [`TiRuntime`](#handle-tiruntime). The user *must not* manipulate multiple [`TiRuntime`](#handle-tiruntime)s in the same thread.

---
### Handle `TiAotModule`

> Stable since Taichi version: 1.4.0

```c
// handle.aot_module
typedef struct TiAotModule_t* TiAotModule;
```

An ahead-of-time (AOT) compiled Taichi module, which contains a collection of kernels and compute graphs.

---
### Handle `TiMemory`

> Stable since Taichi version: 1.4.0

```c
// handle.memory
typedef struct TiMemory_t* TiMemory;
```

A contiguous allocation of device memory.

---
### Handle `TiImage`

> Stable since Taichi version: 1.4.0

```c
// handle.image
typedef struct TiImage_t* TiImage;
```

A contiguous allocation of device image.

---
### Handle `TiKernel`

> Stable since Taichi version: 1.4.0

```c
// handle.kernel
typedef struct TiKernel_t* TiKernel;
```

A Taichi kernel that can be launched on the offload target for execution.

---
### Handle `TiComputeGraph`

> Stable since Taichi version: 1.4.0

```c
// handle.compute_graph
typedef struct TiComputeGraph_t* TiComputeGraph;
```

A collection of Taichi kernels (a compute graph) to launch on the offload target in a predefined order.

---
### Enumeration `TiError`

> Stable since Taichi version: 1.4.0

```c
// enumeration.error
typedef enum TiError {
  TI_ERROR_SUCCESS = 0,
  TI_ERROR_NOT_SUPPORTED = -1,
  TI_ERROR_CORRUPTED_DATA = -2,
  TI_ERROR_NAME_NOT_FOUND = -3,
  TI_ERROR_INVALID_ARGUMENT = -4,
  TI_ERROR_ARGUMENT_NULL = -5,
  TI_ERROR_ARGUMENT_OUT_OF_RANGE = -6,
  TI_ERROR_ARGUMENT_NOT_FOUND = -7,
  TI_ERROR_INVALID_INTEROP = -8,
  TI_ERROR_INVALID_STATE = -9,
  TI_ERROR_INCOMPATIBLE_MODULE = -10,
  TI_ERROR_OUT_OF_MEMORY = -11,
  TI_ERROR_MAX_ENUM = 0xffffffff,
} TiError;
```

Errors reported by the Taichi C-API.

- `TI_ERROR_SUCCESS`: The Taichi C-API invocation finished gracefully.
- `TI_ERROR_NOT_SUPPORTED`: The invoked API, or the combination of parameters is not supported by the Taichi C-API.
- `TI_ERROR_CORRUPTED_DATA`: Provided data is corrupted.
- `TI_ERROR_NAME_NOT_FOUND`: Provided name does not refer to any existing item.
- `TI_ERROR_INVALID_ARGUMENT`: One or more function arguments violate constraints specified in C-API documents, or kernel arguments mismatch the kernel argument list defined in the AOT module.
- `TI_ERROR_ARGUMENT_NULL`: One or more by-reference (pointer) function arguments point to null.
- `TI_ERROR_ARGUMENT_OUT_OF_RANGE`: One or more function arguments are out of its acceptable range; or enumeration arguments have undefined value.
- `TI_ERROR_ARGUMENT_NOT_FOUND`: One or more kernel arguments are missing.
- `TI_ERROR_INVALID_INTEROP`: The intended interoperation is not possible on the current arch. For example, attempts to export a Vulkan object from a CUDA runtime are not allowed.
- `TI_ERROR_INVALID_STATE`: The Taichi C-API enters an unrecoverable invalid state. Related Taichi objects are potentially corrupted. The users *should* release the contaminated resources for stability. Please feel free to file an issue if you encountered this error in a normal routine.
- `TI_ERROR_INCOMPATIBLE_MODULE`: The AOT module is not compatible with the current runtime.

---
### Enumeration `TiArch`

> Stable since Taichi version: 1.4.0

```c
// enumeration.arch
typedef enum TiArch {
  TI_ARCH_RESERVED = 0,
  TI_ARCH_VULKAN = 1,
  TI_ARCH_METAL = 2,
  TI_ARCH_CUDA = 3,
  TI_ARCH_X64 = 4,
  TI_ARCH_ARM64 = 5,
  TI_ARCH_OPENGL = 6,
  TI_ARCH_GLES = 7,
  TI_ARCH_MAX_ENUM = 0xffffffff,
} TiArch;
```

Types of backend archs.

- `TI_ARCH_VULKAN`: Vulkan GPU backend.
- `TI_ARCH_METAL`: Metal GPU backend.
- `TI_ARCH_CUDA`: NVIDIA CUDA GPU backend.
- `TI_ARCH_X64`: x64 native CPU backend.
- `TI_ARCH_ARM64`: Arm64 native CPU backend.
- `TI_ARCH_OPENGL`: OpenGL GPU backend.
- `TI_ARCH_GLES`: OpenGL ES GPU backend.

---
### Enumeration `TiCapability`

> Stable since Taichi version: 1.4.0

```c
// enumeration.capability
typedef enum TiCapability {
  TI_CAPABILITY_RESERVED = 0,
  TI_CAPABILITY_SPIRV_VERSION = 1,
  TI_CAPABILITY_SPIRV_HAS_INT8 = 2,
  TI_CAPABILITY_SPIRV_HAS_INT16 = 3,
  TI_CAPABILITY_SPIRV_HAS_INT64 = 4,
  TI_CAPABILITY_SPIRV_HAS_FLOAT16 = 5,
  TI_CAPABILITY_SPIRV_HAS_FLOAT64 = 6,
  TI_CAPABILITY_SPIRV_HAS_ATOMIC_INT64 = 7,
  TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16 = 8,
  TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16_ADD = 9,
  TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16_MINMAX = 10,
  TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT = 11,
  TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT_ADD = 12,
  TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT_MINMAX = 13,
  TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64 = 14,
  TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64_ADD = 15,
  TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64_MINMAX = 16,
  TI_CAPABILITY_SPIRV_HAS_VARIABLE_PTR = 17,
  TI_CAPABILITY_SPIRV_HAS_PHYSICAL_STORAGE_BUFFER = 18,
  TI_CAPABILITY_SPIRV_HAS_SUBGROUP_BASIC = 19,
  TI_CAPABILITY_SPIRV_HAS_SUBGROUP_VOTE = 20,
  TI_CAPABILITY_SPIRV_HAS_SUBGROUP_ARITHMETIC = 21,
  TI_CAPABILITY_SPIRV_HAS_SUBGROUP_BALLOT = 22,
  TI_CAPABILITY_SPIRV_HAS_NON_SEMANTIC_INFO = 23,
  TI_CAPABILITY_SPIRV_HAS_NO_INTEGER_WRAP_DECORATION = 24,
  TI_CAPABILITY_MAX_ENUM = 0xffffffff,
} TiCapability;
```

Device capabilities.

---
### Structure `TiCapabilityLevelInfo`

> Stable since Taichi version: 1.4.0

```c
// structure.capability_level_info
typedef struct TiCapabilityLevelInfo {
  TiCapability capability;
  uint32_t level;
} TiCapabilityLevelInfo;
```

An integral device capability level. It currently is not guaranteed that a higher level value is compatible with a lower level value.

---
### Enumeration `TiDataType`

> Stable since Taichi version: 1.4.0

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

Elementary (primitive) data types. There might be vendor-specific constraints on the available data types so it's recommended to use 32-bit data types if multi-platform distribution is desired.

- `TI_DATA_TYPE_F16`: 16-bit IEEE 754 half-precision floating-point number.
- `TI_DATA_TYPE_F32`: 32-bit IEEE 754 single-precision floating-point number.
- `TI_DATA_TYPE_F64`: 64-bit IEEE 754 double-precision floating-point number.
- `TI_DATA_TYPE_I8`: 8-bit one's complement signed integer.
- `TI_DATA_TYPE_I16`: 16-bit one's complement signed integer.
- `TI_DATA_TYPE_I32`: 32-bit one's complement signed integer.
- `TI_DATA_TYPE_I64`: 64-bit one's complement signed integer.
- `TI_DATA_TYPE_U8`: 8-bit unsigned integer.
- `TI_DATA_TYPE_U16`: 16-bit unsigned integer.
- `TI_DATA_TYPE_U32`: 32-bit unsigned integer.
- `TI_DATA_TYPE_U64`: 64-bit unsigned integer.

---
### Enumeration `TiArgumentType`

> Stable since Taichi version: 1.4.0

```c
// enumeration.argument_type
typedef enum TiArgumentType {
  TI_ARGUMENT_TYPE_I32 = 0,
  TI_ARGUMENT_TYPE_F32 = 1,
  TI_ARGUMENT_TYPE_NDARRAY = 2,
  TI_ARGUMENT_TYPE_TEXTURE = 3,
  TI_ARGUMENT_TYPE_SCALAR = 4,
  TI_ARGUMENT_TYPE_TENSOR = 5,
  TI_ARGUMENT_TYPE_MAX_ENUM = 0xffffffff,
} TiArgumentType;
```

Types of kernel and compute graph argument.

- `TI_ARGUMENT_TYPE_I32`: 32-bit one's complement signed integer.
- `TI_ARGUMENT_TYPE_F32`: 32-bit IEEE 754 single-precision floating-point number.
- `TI_ARGUMENT_TYPE_NDARRAY`: ND-array wrapped around a [`TiMemory`](#handle-timemory).
- `TI_ARGUMENT_TYPE_TEXTURE`: Texture wrapped around a [`TiImage`](#handle-tiimage).
- `TI_ARGUMENT_TYPE_SCALAR`: Typed scalar.
- `TI_ARGUMENT_TYPE_TENSOR`: Typed tensor.


---
### BitField `TiMemoryUsageFlags`

> Stable since Taichi version: 1.4.0

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

Usages of a memory allocation. Taichi requires kernel argument memories to be allocated with `TI_MEMORY_USAGE_STORAGE_BIT`.

- `TI_MEMORY_USAGE_STORAGE_BIT`: The memory can be read/write accessed by any kernel.
- `TI_MEMORY_USAGE_UNIFORM_BIT`: The memory can be used as a uniform buffer in graphics pipelines.
- `TI_MEMORY_USAGE_VERTEX_BIT`: The memory can be used as a vertex buffer in graphics pipelines.
- `TI_MEMORY_USAGE_INDEX_BIT`: The memory can be used as an index buffer in graphics pipelines.

---
### Structure `TiMemoryAllocateInfo`

> Stable since Taichi version: 1.4.0

```c
// structure.memory_allocate_info
typedef struct TiMemoryAllocateInfo {
  uint64_t size;
  TiBool host_write;
  TiBool host_read;
  TiBool export_sharing;
  TiMemoryUsageFlags usage;
} TiMemoryAllocateInfo;
```

Parameters of a newly allocated memory.

- `size`: Size of the allocation in bytes.
- `host_write`: True if the host needs to write to the allocated memory.
- `host_read`: True if the host needs to read from the allocated memory.
- `export_sharing`: True if the memory allocation needs to be exported to other backends (e.g., from Vulkan to CUDA).
- `usage`: All possible usage of this memory allocation. In most cases, `TI_MEMORY_USAGE_STORAGE_BIT` is enough.

---
### Structure `TiMemorySlice`

> Stable since Taichi version: 1.4.0

```c
// structure.memory_slice
typedef struct TiMemorySlice {
  TiMemory memory;
  uint64_t offset;
  uint64_t size;
} TiMemorySlice;
```

A subsection of a memory allocation. The sum of `offset` and `size` cannot exceed the size of `memory`.

- `memory`: The subsectioned memory allocation.
- `offset`: Offset from the beginning of the allocation.
- `size`: Size of the subsection.

---
### Structure `TiNdShape`

> Stable since Taichi version: 1.4.0

```c
// structure.nd_shape
typedef struct TiNdShape {
  uint32_t dim_count;
  uint32_t dims[16];
} TiNdShape;
```

Multi-dimensional size of an ND-array. Dimension sizes after `dim_count` are ignored.

- `dim_count`: Number of dimensions.
- `dims`: Dimension sizes.

---
### Structure `TiNdArray`

> Stable since Taichi version: 1.4.0

```c
// structure.nd_array
typedef struct TiNdArray {
  TiMemory memory;
  TiNdShape shape;
  TiNdShape elem_shape;
  TiDataType elem_type;
} TiNdArray;
```

Multi-dimensional array of dense primitive data.

- `memory`: Memory bound to the ND-array.
- `shape`: Shape of the ND-array.
- `elem_shape`: Shape of the ND-array elements. It *must not* be empty for vector or matrix ND-arrays.
- `elem_type`: Primitive data type of the ND-array elements.

---
### BitField `TiImageUsageFlags`

> Stable since Taichi version: 1.4.0

```c
// bit_field.image_usage
typedef enum TiImageUsageFlagBits {
  TI_IMAGE_USAGE_STORAGE_BIT = 1 << 0,
  TI_IMAGE_USAGE_SAMPLED_BIT = 1 << 1,
  TI_IMAGE_USAGE_ATTACHMENT_BIT = 1 << 2,
} TiImageUsageFlagBits;
typedef TiFlags TiImageUsageFlags;
```

Usages of an image allocation. Taichi requires kernel argument images to be allocated with `TI_IMAGE_USAGE_STORAGE_BIT` and `TI_IMAGE_USAGE_SAMPLED_BIT`.

- `TI_IMAGE_USAGE_STORAGE_BIT`: The image can be read/write accessed by any kernel.
- `TI_IMAGE_USAGE_SAMPLED_BIT`: The image can be read-only accessed by any kernel.
- `TI_IMAGE_USAGE_ATTACHMENT_BIT`: The image can be used as a color or depth-stencil attachment depending on its format.

---
### Enumeration `TiImageDimension`

> Stable since Taichi version: 1.4.0

```c
// enumeration.image_dimension
typedef enum TiImageDimension {
  TI_IMAGE_DIMENSION_1D = 0,
  TI_IMAGE_DIMENSION_2D = 1,
  TI_IMAGE_DIMENSION_3D = 2,
  TI_IMAGE_DIMENSION_1D_ARRAY = 3,
  TI_IMAGE_DIMENSION_2D_ARRAY = 4,
  TI_IMAGE_DIMENSION_CUBE = 5,
  TI_IMAGE_DIMENSION_MAX_ENUM = 0xffffffff,
} TiImageDimension;
```

Dimensions of an image allocation.

- `TI_IMAGE_DIMENSION_1D`: The image is 1-dimensional.
- `TI_IMAGE_DIMENSION_2D`: The image is 2-dimensional.
- `TI_IMAGE_DIMENSION_3D`: The image is 3-dimensional.
- `TI_IMAGE_DIMENSION_1D_ARRAY`: The image is 1-dimensional and it has one or more layers.
- `TI_IMAGE_DIMENSION_2D_ARRAY`: The image is 2-dimensional and it has one or more layers.
- `TI_IMAGE_DIMENSION_CUBE`: The image is 2-dimensional and it has 6 layers for the faces towards +X, -X, +Y, -Y, +Z, -Z in sequence.

---
### Enumeration `TiImageLayout`

> Stable since Taichi version: 1.4.0

```c
// enumeration.image_layout
typedef enum TiImageLayout {
  TI_IMAGE_LAYOUT_UNDEFINED = 0,
  TI_IMAGE_LAYOUT_SHADER_READ = 1,
  TI_IMAGE_LAYOUT_SHADER_WRITE = 2,
  TI_IMAGE_LAYOUT_SHADER_READ_WRITE = 3,
  TI_IMAGE_LAYOUT_COLOR_ATTACHMENT = 4,
  TI_IMAGE_LAYOUT_COLOR_ATTACHMENT_READ = 5,
  TI_IMAGE_LAYOUT_DEPTH_ATTACHMENT = 6,
  TI_IMAGE_LAYOUT_DEPTH_ATTACHMENT_READ = 7,
  TI_IMAGE_LAYOUT_TRANSFER_DST = 8,
  TI_IMAGE_LAYOUT_TRANSFER_SRC = 9,
  TI_IMAGE_LAYOUT_PRESENT_SRC = 10,
  TI_IMAGE_LAYOUT_MAX_ENUM = 0xffffffff,
} TiImageLayout;
```

- `TI_IMAGE_LAYOUT_UNDEFINED`: Undefined layout. An image in this layout does not contain any semantical information.
- `TI_IMAGE_LAYOUT_SHADER_READ`: Optimal layout for read-only access, including sampling.
- `TI_IMAGE_LAYOUT_SHADER_WRITE`: Optimal layout for write-only access.
- `TI_IMAGE_LAYOUT_SHADER_READ_WRITE`: Optimal layout for read/write access.
- `TI_IMAGE_LAYOUT_COLOR_ATTACHMENT`: Optimal layout as a color attachment.
- `TI_IMAGE_LAYOUT_COLOR_ATTACHMENT_READ`: Optimal layout as an input color attachment.
- `TI_IMAGE_LAYOUT_DEPTH_ATTACHMENT`: Optimal layout as a depth attachment.
- `TI_IMAGE_LAYOUT_DEPTH_ATTACHMENT_READ`: Optimal layout as an input depth attachment.
- `TI_IMAGE_LAYOUT_TRANSFER_DST`: Optimal layout as a data copy destination.
- `TI_IMAGE_LAYOUT_TRANSFER_SRC`: Optimal layout as a data copy source.
- `TI_IMAGE_LAYOUT_PRESENT_SRC`:  Optimal layout as a presentation source.

---
### Enumeration `TiFormat`

> Stable since Taichi version: 1.4.0

```c
// enumeration.format
typedef enum TiFormat {
  TI_FORMAT_UNKNOWN = 0,
  TI_FORMAT_R8 = 1,
  TI_FORMAT_RG8 = 2,
  TI_FORMAT_RGBA8 = 3,
  TI_FORMAT_RGBA8SRGB = 4,
  TI_FORMAT_BGRA8 = 5,
  TI_FORMAT_BGRA8SRGB = 6,
  TI_FORMAT_R8U = 7,
  TI_FORMAT_RG8U = 8,
  TI_FORMAT_RGBA8U = 9,
  TI_FORMAT_R8I = 10,
  TI_FORMAT_RG8I = 11,
  TI_FORMAT_RGBA8I = 12,
  TI_FORMAT_R16 = 13,
  TI_FORMAT_RG16 = 14,
  TI_FORMAT_RGB16 = 15,
  TI_FORMAT_RGBA16 = 16,
  TI_FORMAT_R16U = 17,
  TI_FORMAT_RG16U = 18,
  TI_FORMAT_RGB16U = 19,
  TI_FORMAT_RGBA16U = 20,
  TI_FORMAT_R16I = 21,
  TI_FORMAT_RG16I = 22,
  TI_FORMAT_RGB16I = 23,
  TI_FORMAT_RGBA16I = 24,
  TI_FORMAT_R16F = 25,
  TI_FORMAT_RG16F = 26,
  TI_FORMAT_RGB16F = 27,
  TI_FORMAT_RGBA16F = 28,
  TI_FORMAT_R32U = 29,
  TI_FORMAT_RG32U = 30,
  TI_FORMAT_RGB32U = 31,
  TI_FORMAT_RGBA32U = 32,
  TI_FORMAT_R32I = 33,
  TI_FORMAT_RG32I = 34,
  TI_FORMAT_RGB32I = 35,
  TI_FORMAT_RGBA32I = 36,
  TI_FORMAT_R32F = 37,
  TI_FORMAT_RG32F = 38,
  TI_FORMAT_RGB32F = 39,
  TI_FORMAT_RGBA32F = 40,
  TI_FORMAT_DEPTH16 = 41,
  TI_FORMAT_DEPTH24STENCIL8 = 42,
  TI_FORMAT_DEPTH32F = 43,
  TI_FORMAT_MAX_ENUM = 0xffffffff,
} TiFormat;
```

Texture formats. The availability of texture formats depends on runtime support.

---
### Structure `TiImageOffset`

> Stable since Taichi version: 1.4.0

```c
// structure.image_offset
typedef struct TiImageOffset {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t array_layer_offset;
} TiImageOffset;
```

Offsets of an image in X, Y, Z, and array layers.

- `x`: Image offset in the X direction.
- `y`: Image offset in the Y direction. *Must* be 0 if the image has a dimension of `TI_IMAGE_DIMENSION_1D` or `TI_IMAGE_DIMENSION_1D_ARRAY`.
- `z`: Image offset in the Z direction. *Must* be 0 if the image has a dimension of `TI_IMAGE_DIMENSION_1D`, `TI_IMAGE_DIMENSION_2D`, `TI_IMAGE_DIMENSION_1D_ARRAY`, `TI_IMAGE_DIMENSION_2D_ARRAY` or `TI_IMAGE_DIMENSION_CUBE_ARRAY`.
- `array_layer_offset`: Image offset in array layers. *Must* be 0 if the image has a dimension of `TI_IMAGE_DIMENSION_1D`, `TI_IMAGE_DIMENSION_2D` or `TI_IMAGE_DIMENSION_3D`.

---
### Structure `TiImageExtent`

> Stable since Taichi version: 1.4.0

```c
// structure.image_extent
typedef struct TiImageExtent {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t array_layer_count;
} TiImageExtent;
```

Extents of an image in X, Y, Z, and array layers.

- `width`: Image extent in the X direction.
- `height`: Image extent in the Y direction. *Must* be 1 if the image has a dimension of `TI_IMAGE_DIMENSION_1D` or `TI_IMAGE_DIMENSION_1D_ARRAY`.
- `depth`: Image extent in the Z direction. *Must* be 1 if the image has a dimension of `TI_IMAGE_DIMENSION_1D`, `TI_IMAGE_DIMENSION_2D`, `TI_IMAGE_DIMENSION_1D_ARRAY`, `TI_IMAGE_DIMENSION_2D_ARRAY` or `TI_IMAGE_DIMENSION_CUBE_ARRAY`.
- `array_layer_count`: Image extent in array layers. *Must* be 1 if the image has a dimension of `TI_IMAGE_DIMENSION_1D`, `TI_IMAGE_DIMENSION_2D` or `TI_IMAGE_DIMENSION_3D`. *Must* be 6 if the image has a dimension of `TI_IMAGE_DIMENSION_CUBE_ARRAY`.

---
### Structure `TiImageAllocateInfo`

> Stable since Taichi version: 1.4.0

```c
// structure.image_allocate_info
typedef struct TiImageAllocateInfo {
  TiImageDimension dimension;
  TiImageExtent extent;
  uint32_t mip_level_count;
  TiFormat format;
  TiBool export_sharing;
  TiImageUsageFlags usage;
} TiImageAllocateInfo;
```

Parameters of a newly allocated image.

- `dimension`: Image dimension.
- `extent`: Image extent.
- `mip_level_count`: Number of mip-levels.
- `format`: Image texel format.
- `export_sharing`: True if the memory allocation needs to be exported to other backends (e.g., from Vulkan to CUDA).
- `usage`: All possible usages of this image allocation. In most cases, `TI_IMAGE_USAGE_STORAGE_BIT` and `TI_IMAGE_USAGE_SAMPLED_BIT` enough.

---
### Structure `TiImageSlice`

> Stable since Taichi version: 1.4.0

```c
// structure.image_slice
typedef struct TiImageSlice {
  TiImage image;
  TiImageOffset offset;
  TiImageExtent extent;
  uint32_t mip_level;
} TiImageSlice;
```

A subsection of a memory allocation. The sum of `offset` and `extent` in each dimension cannot exceed the size of `image`.

- `image`: The subsectioned image allocation.
- `offset`: Offset from the beginning of the allocation in each dimension.
- `extent`: Size of the subsection in each dimension.
- `mip_level`: The subsectioned mip-level.

---
### Structure `TiTexture`

> Stable since Taichi version: 1.4.0

```c
// structure.texture
typedef struct TiTexture {
  TiImage image;
  TiSampler sampler;
  TiImageDimension dimension;
  TiImageExtent extent;
  TiFormat format;
} TiTexture;
```

Image data bound to a sampler.

- `image`: Image bound to the texture.
- `sampler`: The bound sampler that controls the sampling behavior of `image`.
- `dimension`: Image Dimension.
- `extent`: Image extent.
- `format`: Image texel format.

---
### Union `TiScalarValue`

> Stable since Taichi version: 1.5.0

```c
// union.scalar_value
typedef union TiScalarValue {
  uint8_t x8;
  uint16_t x16;
  uint32_t x32;
  uint64_t x64;
} TiScalarValue;
```

Scalar value represented by a power-of-two number of bits.

**NOTE** The unsigned integer types merely hold the number of bits in memory and doesn't reflect any type of the underlying data. For example, a 32-bit floating-point scalar value is assigned by `*(float*)&scalar_value.x32 = 0.0f`; a 16-bit signed integer is assigned by `*(int16_t)&scalar_vaue.x16 = 1`. The actual type of the scalar is hinted via `type`.

- `x8`: Scalar value that fits into 8 bits.
- `x16`: Scalar value that fits into 16 bits.
- `x32`: Scalar value that fits into 32 bits.
- `x64`: Scalar value that fits into 64 bits.

---
### Structure `TiScalar`

> Stable since Taichi version: 1.5.0

```c
// structure.scalar
typedef struct TiScalar {
  TiDataType type;
  TiScalarValue value;
} TiScalar;
```

A typed scalar value.

---
### Union `TiArgumentValue`

> Stable since Taichi version: 1.4.0

```c
// union.argument_value
typedef union TiArgumentValue {
  int32_t i32;
  float f32;
  TiNdArray ndarray;
  TiTexture texture;
  TiScalar scalar;
  TiTensor tensor;
} TiArgumentValue;
```

A scalar or structured argument value.

- `i32`: Value of a 32-bit one's complement signed integer. This is equivalent to `x32` with `TI_DATA_TYPE_I32`.
- `f32`: Value of a 32-bit IEEE 754 single-precision floating-poing number. This is equivalent to `x32` with `TI_DATA_TYPE_F32`.
- `ndarray`: An ND-array to be bound.
- `texture`: A texture to be bound.
- `scalar`: An scalar to be bound.
- `tensor`: A tensor to be bound.

---
### Structure `TiArgument`

> Stable since Taichi version: 1.4.0

```c
// structure.argument
typedef struct TiArgument {
  TiArgumentType type;
  TiArgumentValue value;
} TiArgument;
```

An argument value to feed kernels.

- `type`: Type of the argument.
- `value`: Value of the argument.

---
### Structure `TiNamedArgument`

> Stable since Taichi version: 1.4.0

```c
// structure.named_argument
typedef struct TiNamedArgument {
  const char* name;
  TiArgument argument;
} TiNamedArgument;
```

A named argument value to feed compute graphs.

- `name`: Name of the argument.
- `argument`: Argument body.

---
### Function `ti_get_version`

> Stable since Taichi version: 1.4.0

```c
// function.get_version
TI_DLL_EXPORT uint32_t TI_API_CALL ti_get_version(
);
```

Get the current taichi version. It has the same value as `TI_C_API_VERSION` as defined in `taichi_core.h`.

---
### Function `ti_get_available_archs`

> Stable since Taichi version: 1.4.0

```c
// function.get_available_archs
TI_DLL_EXPORT void TI_API_CALL ti_get_available_archs(
  uint32_t* arch_count,
  TiArch* archs
);
```

Gets a list of available archs on the current platform. An arch is only available if:

1. The Runtime library is compiled with its support;
2. The current platform is installed with a capable hardware or an emulation software.

An available arch has at least one device available, i.e., device index 0 is always available. If an arch is not available on the current platform, a call to [`ti_create_runtime`](#function-ti_create_runtime) with that arch is guaranteed failing.

**WARNING** Please also note that the order or returned archs is *undefined*.

---
### Function `ti_get_last_error`

> Stable since Taichi version: 1.4.0

```c
// function.get_last_error
TI_DLL_EXPORT TiError TI_API_CALL ti_get_last_error(
  uint64_t* message_size,
  char* message
);
```

Gets the last error raised by Taichi C-API invocations. Returns the semantical error code.

- `message_size`: Size of textual error message in `message`
- `message`: Text buffer for the textual error message. Ignored when `message_size` is 0.

---
### Function `ti_set_last_error`

> Stable since Taichi version: 1.4.0

```c
// function.set_last_error
TI_DLL_EXPORT void TI_API_CALL ti_set_last_error(
  TiError error,
  const char* message
);
```

Sets the provided error as the last error raised by Taichi C-API invocations. It can be useful in extended validation procedures in Taichi C-API wrappers and helper libraries.

- `error`: Semantical error code.
- `message`: A null-terminated string of the textual error message or `nullptr` for empty error message.

---
### Function `ti_create_runtime`

> Stable since Taichi version: 1.4.0

```c
// function.create_runtime
TI_DLL_EXPORT TiRuntime TI_API_CALL ti_create_runtime(
  TiArch arch,
  uint32_t device_index
);
```

Creates a Taichi Runtime with the specified [`TiArch`](#enumeration-tiarch).

- `arch`: Arch of Taichi Runtime.
- `device_index`: The index of device in `arch` to create Taichi Runtime on.

---
### Function `ti_destroy_runtime`

> Stable since Taichi version: 1.4.0

```c
// function.destroy_runtime
TI_DLL_EXPORT void TI_API_CALL ti_destroy_runtime(
  TiRuntime runtime
);
```

Destroys a Taichi Runtime.

---
### Function `ti_set_runtime_capabilities_ext`

> Stable since Taichi version: 1.4.0

```c
// function.set_runtime_capabilities
TI_DLL_EXPORT void TI_API_CALL ti_set_runtime_capabilities_ext(
  TiRuntime runtime,
  uint32_t capability_count,
  const TiCapabilityLevelInfo* capabilities
);
```

Force override the list of available capabilities in the runtime instance.

---
### Function `ti_get_runtime_capabilities`

> Stable since Taichi version: 1.4.0

```c
// function.get_runtime_capabilities
TI_DLL_EXPORT void TI_API_CALL ti_get_runtime_capabilities(
  TiRuntime runtime,
  uint32_t* capability_count,
  TiCapabilityLevelInfo* capabilities
);
```

Gets all capabilities available on the runtime instance.

- `capability_count`: The total number of capabilities available.
- `capabilities`: Returned capabilities.

---
### Function `ti_allocate_memory`

> Stable since Taichi version: 1.4.0

```c
// function.allocate_memory
TI_DLL_EXPORT TiMemory TI_API_CALL ti_allocate_memory(
  TiRuntime runtime,
  const TiMemoryAllocateInfo* allocate_info
);
```

Allocates a contiguous device memory with provided parameters.

---
### Function `ti_free_memory`

> Stable since Taichi version: 1.4.0

```c
// function.free_memory
TI_DLL_EXPORT void TI_API_CALL ti_free_memory(
  TiRuntime runtime,
  TiMemory memory
);
```

Frees a memory allocation.

---
### Function `ti_map_memory`

> Stable since Taichi version: 1.4.0

```c
// function.map_memory
TI_DLL_EXPORT void* TI_API_CALL ti_map_memory(
  TiRuntime runtime,
  TiMemory memory
);
```

Maps a device memory to a host-addressable space. You *must* ensure that the device is not being used by any device command before the mapping.

---
### Function `ti_unmap_memory`

> Stable since Taichi version: 1.4.0

```c
// function.unmap_memory
TI_DLL_EXPORT void TI_API_CALL ti_unmap_memory(
  TiRuntime runtime,
  TiMemory memory
);
```

Unmaps a device memory and makes any host-side changes about the memory visible to the device. You *must* ensure that there is no further access to the previously mapped host-addressable space.

---
### Function `ti_allocate_image`

> Stable since Taichi version: 1.4.0

```c
// function.allocate_image
TI_DLL_EXPORT TiImage TI_API_CALL ti_allocate_image(
  TiRuntime runtime,
  const TiImageAllocateInfo* allocate_info
);
```

Allocates a device image with provided parameters.

---
### Function `ti_free_image`

> Stable since Taichi version: 1.4.0

```c
// function.free_image
TI_DLL_EXPORT void TI_API_CALL ti_free_image(
  TiRuntime runtime,
  TiImage image
);
```

Frees an image allocation.

---
### Function `ti_copy_memory_device_to_device` (Device Command)

> Stable since Taichi version: 1.4.0

```c
// function.copy_memory_device_to_device
TI_DLL_EXPORT void TI_API_CALL ti_copy_memory_device_to_device(
  TiRuntime runtime,
  const TiMemorySlice* dst_memory,
  const TiMemorySlice* src_memory
);
```

Copies the data in a contiguous subsection of the device memory to another subsection. The two subsections *must not* overlap.

---
### Function `ti_track_image_ext`

> Stable since Taichi version: 1.4.0

```c
// function.track_image
TI_DLL_EXPORT void TI_API_CALL ti_track_image_ext(
  TiRuntime runtime,
  TiImage image,
  TiImageLayout layout
);
```

Tracks the device image with the provided image layout. Because Taichi tracks image layouts internally, it is *only* useful to inform Taichi that the image is transitioned to a new layout by external procedures.

---
### Function `ti_transition_image` (Device Command)

> Stable since Taichi version: 1.4.0

```c
// function.transition_image
TI_DLL_EXPORT void TI_API_CALL ti_transition_image(
  TiRuntime runtime,
  TiImage image,
  TiImageLayout layout
);
```

Transitions the image to the provided image layout. Because Taichi tracks image layouts internally, it is *only* useful to enforce an image layout for external procedures to use.

---
### Function `ti_launch_kernel` (Device Command)

> Stable since Taichi version: 1.4.0

```c
// function.launch_kernel
TI_DLL_EXPORT void TI_API_CALL ti_launch_kernel(
  TiRuntime runtime,
  TiKernel kernel,
  uint32_t arg_count,
  const TiArgument* args
);
```

Launches a Taichi kernel with the provided arguments. The arguments *must* have the same count and types in the same order as in the source code.

---
### Function `ti_launch_compute_graph` (Device Command)

> Stable since Taichi version: 1.4.0

```c
// function.launch_compute_graph
TI_DLL_EXPORT void TI_API_CALL ti_launch_compute_graph(
  TiRuntime runtime,
  TiComputeGraph compute_graph,
  uint32_t arg_count,
  const TiNamedArgument* args
);
```

Launches a Taichi compute graph with provided named arguments. The named arguments *must* have the same count, names, and types as in the source code.

---
### Function `ti_flush`

> Stable since Taichi version: 1.4.0

```c
// function.flush
TI_DLL_EXPORT void TI_API_CALL ti_flush(
  TiRuntime runtime
);
```

Submits all previously invoked device commands to the offload device for execution.

---
### Function `ti_wait`

> Stable since Taichi version: 1.4.0

```c
// function.wait
TI_DLL_EXPORT void TI_API_CALL ti_wait(
  TiRuntime runtime
);
```

Waits until all previously invoked device commands are executed. Any invoked command that has not been submitted is submitted first.

---
### Function `ti_load_aot_module`

> Stable since Taichi version: 1.4.0

```c
// function.load_aot_module
TI_DLL_EXPORT TiAotModule TI_API_CALL ti_load_aot_module(
  TiRuntime runtime,
  const char* module_path
);
```

Loads a pre-compiled AOT module from the file system.
Returns [`TI_NULL_HANDLE`](#definition-ti_null_handle) if the runtime fails to load the AOT module from the specified path.

---
### Function `ti_create_aot_module`

> Stable since Taichi version: 1.4.0

```c
// function.create_aot_module
TI_DLL_EXPORT TiAotModule TI_API_CALL ti_create_aot_module(
  TiRuntime runtime,
  const void* tcm,
  uint64_t size
);
```

Creates a pre-compiled AOT module from TCM data.
Returns [`TI_NULL_HANDLE`](#definition-ti_null_handle) if the runtime fails to create the AOT module from TCM data.

---
### Function `ti_destroy_aot_module`

> Stable since Taichi version: 1.4.0

```c
// function.destroy_aot_module
TI_DLL_EXPORT void TI_API_CALL ti_destroy_aot_module(
  TiAotModule aot_module
);
```

Destroys a loaded AOT module and releases all related resources.

---
### Function `ti_get_aot_module_kernel`

> Stable since Taichi version: 1.4.0

```c
// function.get_aot_module_kernel
TI_DLL_EXPORT TiKernel TI_API_CALL ti_get_aot_module_kernel(
  TiAotModule aot_module,
  const char* name
);
```

Retrieves a pre-compiled Taichi kernel from the AOT module.
Returns [`TI_NULL_HANDLE`](#definition-ti_null_handle) if the module does not have a kernel of the specified name.

---
### Function `ti_get_aot_module_compute_graph`

> Stable since Taichi version: 1.4.0

```c
// function.get_aot_module_compute_graph
TI_DLL_EXPORT TiComputeGraph TI_API_CALL ti_get_aot_module_compute_graph(
  TiAotModule aot_module,
  const char* name
);
```

Retrieves a pre-compiled compute graph from the AOT module.
Returns [`TI_NULL_HANDLE`](#definition-ti_null_handle) if the module does not have a compute graph of the specified name.
