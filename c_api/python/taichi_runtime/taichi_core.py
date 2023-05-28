"""
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

"""
import ctypes

def load_taichi_c_api() -> ctypes.CDLL:
    import ctypes.util as ctypes_util
    from os import environ
    from pathlib import Path

    path = ctypes_util.find_library('taichi_c_api')

    if path is None:
        taichi_c_api_install_dir = environ['TAICHI_C_API_INSTALL_DIR']
        if taichi_c_api_install_dir != None:
            candidate_file_names = [
                'bin/taichi_c_api.dll',
                'lib/libtaichi_c_api.so',
                'lib/libtaichi_c_api.dylib',
            ]
            taichi_c_api_install_dir = Path(taichi_c_api_install_dir)
            for candidate_file_name in candidate_file_names:
                candidate_file_path = taichi_c_api_install_dir / candidate_file_name
                if candidate_file_path.exists():
                    path = str(candidate_file_path)
                    break

    if path is None:
        raise RuntimeError(
            'Cannot find taichi_c_api. Please set TAICHI_C_API_INSTALL_DIR environment variable.'
        )

    print(f'Found taichi_c_api at {path}')
    out = ctypes.CDLL(path, ctypes.RTLD_LOCAL)
    return out

_LIB = load_taichi_c_api()


TI_C_API_VERSION = 1007000


"""
Alias `TiBool` (1.4.0)

A boolean value. Can be either [`TI_TRUE`](#definition-ti_true) or [`TI_FALSE`](#definition-ti_false). Assignment with other values could lead to undefined behavior.
"""
TiBool = ctypes.c_uint32


"""
Definition `TI_FALSE` (1.4.0)

A condition or a predicate is not satisfied; a statement is invalid.
"""
TI_FALSE = 0


"""
Definition `TI_TRUE` (1.4.0)

A condition or a predicate is satisfied; a statement is valid.
"""
TI_TRUE = 1


"""
Alias `TiFlags` (1.4.0)

A bit field that can be used to represent 32 orthogonal flags. Bits unspecified in the corresponding flag enum are ignored.

> Enumerations and bit-field flags in the C-API have a `TI_XXX_MAX_ENUM` case to ensure the enum has a 32-bit range and in-memory size. It has no semantical impact and can be safely ignored.
"""
TiFlags = ctypes.c_uint32


"""
Definition `TI_NULL_HANDLE` (1.4.0)

A sentinal invalid handle that will never be produced from a valid call to Taichi C-API.
"""
TI_NULL_HANDLE = 0


"""
Handle `TiRuntime` (1.4.0)

Taichi runtime represents an instance of a logical backend and its internal dynamic state. The user is responsible to synchronize any use of [`TiRuntime`](#handle-tiruntime). The user *must not* manipulate multiple [`TiRuntime`](#handle-tiruntime)s in the same thread.
"""
TiRuntime = ctypes.c_void_p


"""
Handle `TiAotModule` (1.4.0)

An ahead-of-time (AOT) compiled Taichi module, which contains a collection of kernels and compute graphs.
"""
TiAotModule = ctypes.c_void_p


"""
Handle `TiMemory` (1.4.0)

A contiguous allocation of device memory.
"""
TiMemory = ctypes.c_void_p


"""
Handle `TiImage` (1.4.0)

A contiguous allocation of device image.
"""
TiImage = ctypes.c_void_p


"""
Handle `TiSampler`

An image sampler. [`TI_NULL_HANDLE`](#definition-ti_null_handle) represents a default image sampler provided by the runtime implementation. The filter modes and address modes of default samplers depend on backend implementation.
"""
TiSampler = ctypes.c_void_p


"""
Handle `TiKernel` (1.4.0)

A Taichi kernel that can be launched on the offload target for execution.
"""
TiKernel = ctypes.c_void_p


"""
Handle `TiComputeGraph` (1.4.0)

A collection of Taichi kernels (a compute graph) to launch on the offload target in a predefined order.
"""
TiComputeGraph = ctypes.c_void_p


"""
Enumeration `TiError` (1.4.0)

Errors reported by the Taichi C-API.
"""
TiError = ctypes.c_int32
# The Taichi C-API invocation finished gracefully.
TI_ERROR_SUCCESS = TiError(0)
# The invoked API, or the combination of parameters is not supported by the Taichi C-API.
TI_ERROR_NOT_SUPPORTED = TiError(-1)
# Provided data is corrupted.
TI_ERROR_CORRUPTED_DATA = TiError(-2)
# Provided name does not refer to any existing item.
TI_ERROR_NAME_NOT_FOUND = TiError(-3)
# One or more function arguments violate constraints specified in C-API documents, or kernel arguments mismatch the kernel argument list defined in the AOT module.
TI_ERROR_INVALID_ARGUMENT = TiError(-4)
# One or more by-reference (pointer) function arguments point to null.
TI_ERROR_ARGUMENT_NULL = TiError(-5)
# One or more function arguments are out of its acceptable range; or enumeration arguments have undefined value.
TI_ERROR_ARGUMENT_OUT_OF_RANGE = TiError(-6)
# One or more kernel arguments are missing.
TI_ERROR_ARGUMENT_NOT_FOUND = TiError(-7)
# The intended interoperation is not possible on the current arch. For example, attempts to export a Vulkan object from a CUDA runtime are not allowed.
TI_ERROR_INVALID_INTEROP = TiError(-8)
# The Taichi C-API enters an unrecoverable invalid state. Related Taichi objects are potentially corrupted. The users *should* release the contaminated resources for stability. Please feel free to file an issue if you encountered this error in a normal routine.
TI_ERROR_INVALID_STATE = TiError(-9)
# The AOT module is not compatible with the current runtime.
TI_ERROR_INCOMPATIBLE_MODULE = TiError(-10)
TI_ERROR_OUT_OF_MEMORY = TiError(-11)
TI_ERROR_MAX_ENUM = TiError(0xffffffff)


"""
Enumeration `TiArch` (1.4.0)

Types of backend archs.
"""
TiArch = ctypes.c_int32
TI_ARCH_RESERVED = TiArch(0)
# Vulkan GPU backend.
TI_ARCH_VULKAN = TiArch(1)
# Metal GPU backend.
TI_ARCH_METAL = TiArch(2)
# NVIDIA CUDA GPU backend.
TI_ARCH_CUDA = TiArch(3)
# x64 native CPU backend.
TI_ARCH_X64 = TiArch(4)
# Arm64 native CPU backend.
TI_ARCH_ARM64 = TiArch(5)
# OpenGL GPU backend.
TI_ARCH_OPENGL = TiArch(6)
# OpenGL ES GPU backend.
TI_ARCH_GLES = TiArch(7)
TI_ARCH_MAX_ENUM = TiArch(0xffffffff)


"""
Enumeration `TiCapability` (1.4.0)

Device capabilities.
"""
TiCapability = ctypes.c_int32
TI_CAPABILITY_RESERVED = TiCapability(0)
TI_CAPABILITY_SPIRV_VERSION = TiCapability(1)
TI_CAPABILITY_SPIRV_HAS_INT8 = TiCapability(2)
TI_CAPABILITY_SPIRV_HAS_INT16 = TiCapability(3)
TI_CAPABILITY_SPIRV_HAS_INT64 = TiCapability(4)
TI_CAPABILITY_SPIRV_HAS_FLOAT16 = TiCapability(5)
TI_CAPABILITY_SPIRV_HAS_FLOAT64 = TiCapability(6)
TI_CAPABILITY_SPIRV_HAS_ATOMIC_INT64 = TiCapability(7)
TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16 = TiCapability(8)
TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16_ADD = TiCapability(9)
TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16_MINMAX = TiCapability(10)
TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT = TiCapability(11)
TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT_ADD = TiCapability(12)
TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT_MINMAX = TiCapability(13)
TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64 = TiCapability(14)
TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64_ADD = TiCapability(15)
TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64_MINMAX = TiCapability(16)
TI_CAPABILITY_SPIRV_HAS_VARIABLE_PTR = TiCapability(17)
TI_CAPABILITY_SPIRV_HAS_PHYSICAL_STORAGE_BUFFER = TiCapability(18)
TI_CAPABILITY_SPIRV_HAS_SUBGROUP_BASIC = TiCapability(19)
TI_CAPABILITY_SPIRV_HAS_SUBGROUP_VOTE = TiCapability(20)
TI_CAPABILITY_SPIRV_HAS_SUBGROUP_ARITHMETIC = TiCapability(21)
TI_CAPABILITY_SPIRV_HAS_SUBGROUP_BALLOT = TiCapability(22)
TI_CAPABILITY_SPIRV_HAS_NON_SEMANTIC_INFO = TiCapability(23)
TI_CAPABILITY_SPIRV_HAS_NO_INTEGER_WRAP_DECORATION = TiCapability(24)
TI_CAPABILITY_MAX_ENUM = TiCapability(0xffffffff)


"""
Structure `TiCapabilityLevelInfo` (1.4.0)

An integral device capability level. It currently is not guaranteed that a higher level value is compatible with a lower level value.
"""
class TiCapabilityLevelInfo(ctypes.Structure): pass
TiCapabilityLevelInfo._fields_ = [
    ('capability', TiCapability),
    ('level', ctypes.c_uint32),
]


"""
Enumeration `TiDataType` (1.4.0)

Elementary (primitive) data types. There might be vendor-specific constraints on the available data types so it's recommended to use 32-bit data types if multi-platform distribution is desired.
"""
TiDataType = ctypes.c_int32
# 16-bit IEEE 754 half-precision floating-point number.
TI_DATA_TYPE_F16 = TiDataType(0)
# 32-bit IEEE 754 single-precision floating-point number.
TI_DATA_TYPE_F32 = TiDataType(1)
# 64-bit IEEE 754 double-precision floating-point number.
TI_DATA_TYPE_F64 = TiDataType(2)
# 8-bit one's complement signed integer.
TI_DATA_TYPE_I8 = TiDataType(3)
# 16-bit one's complement signed integer.
TI_DATA_TYPE_I16 = TiDataType(4)
# 32-bit one's complement signed integer.
TI_DATA_TYPE_I32 = TiDataType(5)
# 64-bit one's complement signed integer.
TI_DATA_TYPE_I64 = TiDataType(6)
TI_DATA_TYPE_U1 = TiDataType(7)
# 8-bit unsigned integer.
TI_DATA_TYPE_U8 = TiDataType(8)
# 16-bit unsigned integer.
TI_DATA_TYPE_U16 = TiDataType(9)
# 32-bit unsigned integer.
TI_DATA_TYPE_U32 = TiDataType(10)
# 64-bit unsigned integer.
TI_DATA_TYPE_U64 = TiDataType(11)
TI_DATA_TYPE_GEN = TiDataType(12)
TI_DATA_TYPE_UNKNOWN = TiDataType(13)
TI_DATA_TYPE_MAX_ENUM = TiDataType(0xffffffff)


"""
Enumeration `TiArgumentType` (1.4.0)

Types of kernel and compute graph argument.
"""
TiArgumentType = ctypes.c_int32
# 32-bit one's complement signed integer.
TI_ARGUMENT_TYPE_I32 = TiArgumentType(0)
# 32-bit IEEE 754 single-precision floating-point number.
TI_ARGUMENT_TYPE_F32 = TiArgumentType(1)
# ND-array wrapped around a `handle.memory`.
TI_ARGUMENT_TYPE_NDARRAY = TiArgumentType(2)
# Texture wrapped around a `handle.image`.
TI_ARGUMENT_TYPE_TEXTURE = TiArgumentType(3)
# Typed scalar.
TI_ARGUMENT_TYPE_SCALAR = TiArgumentType(4)
# Typed tensor.
TI_ARGUMENT_TYPE_TENSOR = TiArgumentType(5)
TI_ARGUMENT_TYPE_MAX_ENUM = TiArgumentType(0xffffffff)


"""
BitField `TiMemoryUsageFlags` (1.4.0)

Usages of a memory allocation. Taichi requires kernel argument memories to be allocated with `TI_MEMORY_USAGE_STORAGE_BIT`.
"""
TiMemoryUsageFlags = TiFlags
TiMemoryUsageFlagBits = ctypes.c_uint32
# The memory can be read/write accessed by any kernel.
TI_MEMORY_USAGE_STORAGE_BIT = TiMemoryUsageFlagBits(1 << 0),
# The memory can be used as a uniform buffer in graphics pipelines.
TI_MEMORY_USAGE_UNIFORM_BIT = TiMemoryUsageFlagBits(1 << 1),
# The memory can be used as a vertex buffer in graphics pipelines.
TI_MEMORY_USAGE_VERTEX_BIT = TiMemoryUsageFlagBits(1 << 2),
# The memory can be used as an index buffer in graphics pipelines.
TI_MEMORY_USAGE_INDEX_BIT = TiMemoryUsageFlagBits(1 << 3),


"""
Structure `TiMemoryAllocateInfo` (1.4.0)

Parameters of a newly allocated memory.
"""
class TiMemoryAllocateInfo(ctypes.Structure): pass
TiMemoryAllocateInfo._fields_ = [
    # Size of the allocation in bytes.
    ('size', ctypes.c_uint64),
    # True if the host needs to write to the allocated memory.
    ('host_write', TiBool),
    # True if the host needs to read from the allocated memory.
    ('host_read', TiBool),
    # True if the memory allocation needs to be exported to other backends (e.g., from Vulkan to CUDA).
    ('export_sharing', TiBool),
    # All possible usage of this memory allocation. In most cases, `bit_field.memory_usage.storage` is enough.
    ('usage', TiMemoryUsageFlags),
]


"""
Structure `TiMemorySlice` (1.4.0)

A subsection of a memory allocation. The sum of `offset` and `size` cannot exceed the size of `memory`.
"""
class TiMemorySlice(ctypes.Structure): pass
TiMemorySlice._fields_ = [
    # The subsectioned memory allocation.
    ('memory', TiMemory),
    # Offset from the beginning of the allocation.
    ('offset', ctypes.c_uint64),
    # Size of the subsection.
    ('size', ctypes.c_uint64),
]


"""
Structure `TiNdShape` (1.4.0)

Multi-dimensional size of an ND-array. Dimension sizes after `dim_count` are ignored.
"""
class TiNdShape(ctypes.Structure): pass
TiNdShape._fields_ = [
    # Number of dimensions.
    ('dim_count', ctypes.c_uint32),
    # Dimension sizes.
    ('dims', ctypes.c_uint32 * 16),
]


"""
Structure `TiNdArray` (1.4.0)

Multi-dimensional array of dense primitive data.
"""
class TiNdArray(ctypes.Structure): pass
TiNdArray._fields_ = [
    # Memory bound to the ND-array.
    ('memory', TiMemory),
    # Shape of the ND-array.
    ('shape', TiNdShape),
    # Shape of the ND-array elements. It *must not* be empty for vector or matrix ND-arrays.
    ('elem_shape', TiNdShape),
    # Primitive data type of the ND-array elements.
    ('elem_type', TiDataType),
]


"""
BitField `TiImageUsageFlags` (1.4.0)

Usages of an image allocation. Taichi requires kernel argument images to be allocated with `TI_IMAGE_USAGE_STORAGE_BIT` and `TI_IMAGE_USAGE_SAMPLED_BIT`.
"""
TiImageUsageFlags = TiFlags
TiImageUsageFlagBits = ctypes.c_uint32
# The image can be read/write accessed by any kernel.
TI_IMAGE_USAGE_STORAGE_BIT = TiImageUsageFlagBits(1 << 0),
# The image can be read-only accessed by any kernel.
TI_IMAGE_USAGE_SAMPLED_BIT = TiImageUsageFlagBits(1 << 1),
# The image can be used as a color or depth-stencil attachment depending on its format.
TI_IMAGE_USAGE_ATTACHMENT_BIT = TiImageUsageFlagBits(1 << 2),


"""
Enumeration `TiImageDimension` (1.4.0)

Dimensions of an image allocation.
"""
TiImageDimension = ctypes.c_int32
# The image is 1-dimensional.
TI_IMAGE_DIMENSION_1D = TiImageDimension(0)
# The image is 2-dimensional.
TI_IMAGE_DIMENSION_2D = TiImageDimension(1)
# The image is 3-dimensional.
TI_IMAGE_DIMENSION_3D = TiImageDimension(2)
# The image is 1-dimensional and it has one or more layers.
TI_IMAGE_DIMENSION_1D_ARRAY = TiImageDimension(3)
# The image is 2-dimensional and it has one or more layers.
TI_IMAGE_DIMENSION_2D_ARRAY = TiImageDimension(4)
# The image is 2-dimensional and it has 6 layers for the faces towards +X, -X, +Y, -Y, +Z, -Z in sequence.
TI_IMAGE_DIMENSION_CUBE = TiImageDimension(5)
TI_IMAGE_DIMENSION_MAX_ENUM = TiImageDimension(0xffffffff)


"""
Enumeration `TiImageLayout` (1.4.0)
"""
TiImageLayout = ctypes.c_int32
# Undefined layout. An image in this layout does not contain any semantical information.
TI_IMAGE_LAYOUT_UNDEFINED = TiImageLayout(0)
# Optimal layout for read-only access, including sampling.
TI_IMAGE_LAYOUT_SHADER_READ = TiImageLayout(1)
# Optimal layout for write-only access.
TI_IMAGE_LAYOUT_SHADER_WRITE = TiImageLayout(2)
# Optimal layout for read/write access.
TI_IMAGE_LAYOUT_SHADER_READ_WRITE = TiImageLayout(3)
# Optimal layout as a color attachment.
TI_IMAGE_LAYOUT_COLOR_ATTACHMENT = TiImageLayout(4)
# Optimal layout as an input color attachment.
TI_IMAGE_LAYOUT_COLOR_ATTACHMENT_READ = TiImageLayout(5)
# Optimal layout as a depth attachment.
TI_IMAGE_LAYOUT_DEPTH_ATTACHMENT = TiImageLayout(6)
# Optimal layout as an input depth attachment.
TI_IMAGE_LAYOUT_DEPTH_ATTACHMENT_READ = TiImageLayout(7)
# Optimal layout as a data copy destination.
TI_IMAGE_LAYOUT_TRANSFER_DST = TiImageLayout(8)
# Optimal layout as a data copy source.
TI_IMAGE_LAYOUT_TRANSFER_SRC = TiImageLayout(9)
# Optimal layout as a presentation source.
TI_IMAGE_LAYOUT_PRESENT_SRC = TiImageLayout(10)
TI_IMAGE_LAYOUT_MAX_ENUM = TiImageLayout(0xffffffff)


"""
Enumeration `TiFormat` (1.4.0)

Texture formats. The availability of texture formats depends on runtime support.
"""
TiFormat = ctypes.c_int32
TI_FORMAT_UNKNOWN = TiFormat(0)
TI_FORMAT_R8 = TiFormat(1)
TI_FORMAT_RG8 = TiFormat(2)
TI_FORMAT_RGBA8 = TiFormat(3)
TI_FORMAT_RGBA8SRGB = TiFormat(4)
TI_FORMAT_BGRA8 = TiFormat(5)
TI_FORMAT_BGRA8SRGB = TiFormat(6)
TI_FORMAT_R8U = TiFormat(7)
TI_FORMAT_RG8U = TiFormat(8)
TI_FORMAT_RGBA8U = TiFormat(9)
TI_FORMAT_R8I = TiFormat(10)
TI_FORMAT_RG8I = TiFormat(11)
TI_FORMAT_RGBA8I = TiFormat(12)
TI_FORMAT_R16 = TiFormat(13)
TI_FORMAT_RG16 = TiFormat(14)
TI_FORMAT_RGB16 = TiFormat(15)
TI_FORMAT_RGBA16 = TiFormat(16)
TI_FORMAT_R16U = TiFormat(17)
TI_FORMAT_RG16U = TiFormat(18)
TI_FORMAT_RGB16U = TiFormat(19)
TI_FORMAT_RGBA16U = TiFormat(20)
TI_FORMAT_R16I = TiFormat(21)
TI_FORMAT_RG16I = TiFormat(22)
TI_FORMAT_RGB16I = TiFormat(23)
TI_FORMAT_RGBA16I = TiFormat(24)
TI_FORMAT_R16F = TiFormat(25)
TI_FORMAT_RG16F = TiFormat(26)
TI_FORMAT_RGB16F = TiFormat(27)
TI_FORMAT_RGBA16F = TiFormat(28)
TI_FORMAT_R32U = TiFormat(29)
TI_FORMAT_RG32U = TiFormat(30)
TI_FORMAT_RGB32U = TiFormat(31)
TI_FORMAT_RGBA32U = TiFormat(32)
TI_FORMAT_R32I = TiFormat(33)
TI_FORMAT_RG32I = TiFormat(34)
TI_FORMAT_RGB32I = TiFormat(35)
TI_FORMAT_RGBA32I = TiFormat(36)
TI_FORMAT_R32F = TiFormat(37)
TI_FORMAT_RG32F = TiFormat(38)
TI_FORMAT_RGB32F = TiFormat(39)
TI_FORMAT_RGBA32F = TiFormat(40)
TI_FORMAT_DEPTH16 = TiFormat(41)
TI_FORMAT_DEPTH24STENCIL8 = TiFormat(42)
TI_FORMAT_DEPTH32F = TiFormat(43)
TI_FORMAT_MAX_ENUM = TiFormat(0xffffffff)


"""
Structure `TiImageOffset` (1.4.0)

Offsets of an image in X, Y, Z, and array layers.
"""
class TiImageOffset(ctypes.Structure): pass
TiImageOffset._fields_ = [
    # Image offset in the X direction.
    ('x', ctypes.c_uint32),
    # Image offset in the Y direction. *Must* be 0 if the image has a dimension of `enumeration.image_dimension.1d` or `enumeration.image_dimension.1d_array`.
    ('y', ctypes.c_uint32),
    # Image offset in the Z direction. *Must* be 0 if the image has a dimension of `enumeration.image_dimension.1d`, `enumeration.image_dimension.2d`, `enumeration.image_dimension.1d_array`, `enumeration.image_dimension.2d_array` or `enumeration.image_dimension.cube_array`.
    ('z', ctypes.c_uint32),
    # Image offset in array layers. *Must* be 0 if the image has a dimension of `enumeration.image_dimension.1d`, `enumeration.image_dimension.2d` or `enumeration.image_dimension.3d`.
    ('array_layer_offset', ctypes.c_uint32),
]


"""
Structure `TiImageExtent` (1.4.0)

Extents of an image in X, Y, Z, and array layers.
"""
class TiImageExtent(ctypes.Structure): pass
TiImageExtent._fields_ = [
    # Image extent in the X direction.
    ('width', ctypes.c_uint32),
    # Image extent in the Y direction. *Must* be 1 if the image has a dimension of `enumeration.image_dimension.1d` or `enumeration.image_dimension.1d_array`.
    ('height', ctypes.c_uint32),
    # Image extent in the Z direction. *Must* be 1 if the image has a dimension of `enumeration.image_dimension.1d`, `enumeration.image_dimension.2d`, `enumeration.image_dimension.1d_array`, `enumeration.image_dimension.2d_array` or `enumeration.image_dimension.cube_array`.
    ('depth', ctypes.c_uint32),
    # Image extent in array layers. *Must* be 1 if the image has a dimension of `enumeration.image_dimension.1d`, `enumeration.image_dimension.2d` or `enumeration.image_dimension.3d`. *Must* be 6 if the image has a dimension of `enumeration.image_dimension.cube_array`.
    ('array_layer_count', ctypes.c_uint32),
]


"""
Structure `TiImageAllocateInfo` (1.4.0)

Parameters of a newly allocated image.
"""
class TiImageAllocateInfo(ctypes.Structure): pass
TiImageAllocateInfo._fields_ = [
    # Image dimension.
    ('dimension', TiImageDimension),
    # Image extent.
    ('extent', TiImageExtent),
    # Number of mip-levels.
    ('mip_level_count', ctypes.c_uint32),
    # Image texel format.
    ('format', TiFormat),
    # True if the memory allocation needs to be exported to other backends (e.g., from Vulkan to CUDA).
    ('export_sharing', TiBool),
    # All possible usages of this image allocation. In most cases, `bit_field.image_usage.storage` and `bit_field.image_usage.sampled` enough.
    ('usage', TiImageUsageFlags),
]


"""
Structure `TiImageSlice` (1.4.0)

A subsection of a memory allocation. The sum of `offset` and `extent` in each dimension cannot exceed the size of `image`.
"""
class TiImageSlice(ctypes.Structure): pass
TiImageSlice._fields_ = [
    # The subsectioned image allocation.
    ('image', TiImage),
    # Offset from the beginning of the allocation in each dimension.
    ('offset', TiImageOffset),
    # Size of the subsection in each dimension.
    ('extent', TiImageExtent),
    # The subsectioned mip-level.
    ('mip_level', ctypes.c_uint32),
]


"""
Enumeration `TiFilter`
"""
TiFilter = ctypes.c_int32
TI_FILTER_NEAREST = TiFilter(0)
TI_FILTER_LINEAR = TiFilter(1)
TI_FILTER_MAX_ENUM = TiFilter(0xffffffff)


"""
Enumeration `TiAddressMode`
"""
TiAddressMode = ctypes.c_int32
TI_ADDRESS_MODE_REPEAT = TiAddressMode(0)
TI_ADDRESS_MODE_MIRRORED_REPEAT = TiAddressMode(1)
TI_ADDRESS_MODE_CLAMP_TO_EDGE = TiAddressMode(2)
TI_ADDRESS_MODE_MAX_ENUM = TiAddressMode(0xffffffff)


"""
Structure `TiSamplerCreateInfo`
"""
class TiSamplerCreateInfo(ctypes.Structure): pass
TiSamplerCreateInfo._fields_ = [
    ('mag_filter', TiFilter),
    ('min_filter', TiFilter),
    ('address_mode', TiAddressMode),
    ('max_anisotropy', ctypes.c_float),
]


"""
Structure `TiTexture` (1.4.0)

Image data bound to a sampler.
"""
class TiTexture(ctypes.Structure): pass
TiTexture._fields_ = [
    # Image bound to the texture.
    ('image', TiImage),
    # The bound sampler that controls the sampling behavior of `structure.texture.image`.
    ('sampler', TiSampler),
    # Image Dimension.
    ('dimension', TiImageDimension),
    # Image extent.
    ('extent', TiImageExtent),
    # Image texel format.
    ('format', TiFormat),
]


"""
Union `TiScalarValue` (1.5.0)

Scalar value represented by a power-of-two number of bits.

**NOTE** The unsigned integer types merely hold the number of bits in memory and doesn't reflect any type of the underlying data. For example, a 32-bit floating-point scalar value is assigned by `*(float*)&scalar_value.x32 = 0.0f`; a 16-bit signed integer is assigned by `*(int16_t)&scalar_vaue.x16 = 1`. The actual type of the scalar is hinted via `type`.
"""
class TiScalarValue(ctypes.Union): pass
TiScalarValue._fields_ = [
    # Scalar value that fits into 8 bits.
    ('x8', ctypes.c_uint8),
    # Scalar value that fits into 16 bits.
    ('x16', ctypes.c_uint16),
    # Scalar value that fits into 32 bits.
    ('x32', ctypes.c_uint32),
    # Scalar value that fits into 64 bits.
    ('x64', ctypes.c_uint64),
]


"""
Structure `TiScalar` (1.5.0)

A typed scalar value.
"""
class TiScalar(ctypes.Structure): pass
TiScalar._fields_ = [
    ('type', TiDataType),
    ('value', TiScalarValue),
]


"""
Union `TiTensorValue`

Tensor value represented by a power-of-two number of bits.
"""
class TiTensorValue(ctypes.Union): pass
TiTensorValue._fields_ = [
    # Tensor value that fits into 8 bits.
    ('x8', ctypes.c_uint8 * 128),
    # Tensor value that fits into 16 bits.
    ('x16', ctypes.c_uint16 * 64),
    # Tensor value that fits into 32 bits.
    ('x32', ctypes.c_uint32 * 32),
    # Tensor value that fits into 64 bits.
    ('x64', ctypes.c_uint64 * 16),
]


"""
Structure `TiTensorValueWithLength`

A tensor value with a length.
"""
class TiTensorValueWithLength(ctypes.Structure): pass
TiTensorValueWithLength._fields_ = [
    ('length', ctypes.c_uint32),
    ('data', TiTensorValue),
]


"""
Structure `TiTensor`

A typed tensor value.
"""
class TiTensor(ctypes.Structure): pass
TiTensor._fields_ = [
    ('type', TiDataType),
    ('contents', TiTensorValueWithLength),
]


"""
Union `TiArgumentValue` (1.4.0)

A scalar or structured argument value.
"""
class TiArgumentValue(ctypes.Union): pass
TiArgumentValue._fields_ = [
    # Value of a 32-bit one's complement signed integer. This is equivalent to `union.scalar_value.x32` with `enumeration.data_type.i32`.
    ('i32', ctypes.c_int32),
    # Value of a 32-bit IEEE 754 single-precision floating-poing number. This is equivalent to `union.scalar_value.x32` with `enumeration.data_type.f32`.
    ('f32', ctypes.c_float),
    # An ND-array to be bound.
    ('ndarray', TiNdArray),
    # A texture to be bound.
    ('texture', TiTexture),
    # An scalar to be bound.
    ('scalar', TiScalar),
    # A tensor to be bound.
    ('tensor', TiTensor),
]


"""
Structure `TiArgument` (1.4.0)

An argument value to feed kernels.
"""
class TiArgument(ctypes.Structure): pass
TiArgument._fields_ = [
    # Type of the argument.
    ('type', TiArgumentType),
    # Value of the argument.
    ('value', TiArgumentValue),
]


"""
Structure `TiNamedArgument` (1.4.0)

A named argument value to feed compute graphs.
"""
class TiNamedArgument(ctypes.Structure): pass
TiNamedArgument._fields_ = [
    # Name of the argument.
    ('name', ctypes.c_void_p),
    # Argument body.
    ('argument', TiArgument),
]


def ti_get_version(
) -> ctypes.c_uint32:
    """
    Function `ti_get_version` (1.4.0)
    
    Get the current taichi version. It has the same value as `TI_C_API_VERSION` as defined in `taichi_core.h`.

    Return value: ctypes.c_uint32
    """
    return _LIB.ti_get_version()


def ti_get_available_archs(
  arch_count: ctypes.c_void_p, # ctypes.c_uint32*,
  archs: ctypes.c_void_p, # TiArch*
) -> None:
    """
    Function `ti_get_available_archs` (1.4.0)
    
    Gets a list of available archs on the current platform. An arch is only available if:
    
    1. The Runtime library is compiled with its support;
    2. The current platform is installed with a capable hardware or an emulation software.
    
    An available arch has at least one device available, i.e., device index 0 is always available. If an arch is not available on the current platform, a call to [`ti_create_runtime`](#function-ti_create_runtime) with that arch is guaranteed failing.
    
    **WARNING** Please also note that the order or returned archs is *undefined*.

    Return value: None

    Parameters:
        arch_count (`ctypes.c_uint32`):
        archs (`TiArch`):
    """
    return _LIB.ti_get_available_archs(arch_count, archs)


def ti_get_last_error(
  message_size: ctypes.c_void_p, # ctypes.c_uint64*,
  message: ctypes.c_void_p, # char*
) -> TiError:
    """
    Function `ti_get_last_error` (1.4.0)
    
    Gets the last error raised by Taichi C-API invocations. Returns the semantical error code.

    Return value: TiError

    Parameters:
        message_size (`ctypes.c_uint64`):
            Size of textual error message in `function.get_last_error.message`
        message (`char`):
            Text buffer for the textual error message. Ignored when `message_size` is 0.
    """
    return _LIB.ti_get_last_error(message_size, message)


def ti_set_last_error(
  error: TiError,
  message: ctypes.c_void_p
) -> None:
    """
    Function `ti_set_last_error` (1.4.0)
    
    Sets the provided error as the last error raised by Taichi C-API invocations. It can be useful in extended validation procedures in Taichi C-API wrappers and helper libraries.

    Return value: None

    Parameters:
        error (`TiError`):
            Semantical error code.
        message (`ctypes.c_void_p`):
            A null-terminated string of the textual error message or `nullptr` for empty error message.
    """
    return _LIB.ti_set_last_error(error, message)


def ti_create_runtime(
  arch: TiArch,
  device_index: ctypes.c_uint32
) -> TiRuntime:
    """
    Function `ti_create_runtime` (1.4.0)
    
    Creates a Taichi Runtime with the specified [`TiArch`](#enumeration-tiarch).

    Return value: TiRuntime

    Parameters:
        arch (`TiArch`):
            Arch of Taichi Runtime.
        device_index (`ctypes.c_uint32`):
            The index of device in `function.create_runtime.arch` to create Taichi Runtime on.
    """
    return _LIB.ti_create_runtime(arch, device_index)


def ti_destroy_runtime(
  runtime: TiRuntime
) -> None:
    """
    Function `ti_destroy_runtime` (1.4.0)
    
    Destroys a Taichi Runtime.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
    """
    return _LIB.ti_destroy_runtime(runtime)


def ti_set_runtime_capabilities_ext(
  runtime: TiRuntime,
  capability_count: ctypes.c_uint32,
  capabilities: ctypes.c_void_p, # const TiCapabilityLevelInfo*
) -> None:
    """
    Function `ti_set_runtime_capabilities_ext` (1.4.0)
    
    Force override the list of available capabilities in the runtime instance.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        capability_count (`ctypes.c_uint32`):
        capabilities (`TiCapabilityLevelInfo`):
    """
    return _LIB.ti_set_runtime_capabilities_ext(runtime, capability_count, capabilities)


def ti_get_runtime_capabilities(
  runtime: TiRuntime,
  capability_count: ctypes.c_void_p, # ctypes.c_uint32*,
  capabilities: ctypes.c_void_p, # TiCapabilityLevelInfo*
) -> None:
    """
    Function `ti_get_runtime_capabilities` (1.4.0)
    
    Gets all capabilities available on the runtime instance.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        capability_count (`ctypes.c_uint32`):
            The total number of capabilities available.
        capabilities (`TiCapabilityLevelInfo`):
            Returned capabilities.
    """
    return _LIB.ti_get_runtime_capabilities(runtime, capability_count, capabilities)


def ti_allocate_memory(
  runtime: TiRuntime,
  allocate_info: ctypes.c_void_p, # const TiMemoryAllocateInfo*
) -> TiMemory:
    """
    Function `ti_allocate_memory` (1.4.0)
    
    Allocates a contiguous device memory with provided parameters.

    Return value: TiMemory

    Parameters:
        runtime (`TiRuntime`):
        allocate_info (`TiMemoryAllocateInfo`):
    """
    return _LIB.ti_allocate_memory(runtime, allocate_info)


def ti_free_memory(
  runtime: TiRuntime,
  memory: TiMemory
) -> None:
    """
    Function `ti_free_memory` (1.4.0)
    
    Frees a memory allocation.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        memory (`TiMemory`):
    """
    return _LIB.ti_free_memory(runtime, memory)


def ti_map_memory(
  runtime: TiRuntime,
  memory: TiMemory
) -> ctypes.c_void_p:
    """
    Function `ti_map_memory` (1.4.0)
    
    Maps a device memory to a host-addressable space. You *must* ensure that the device is not being used by any device command before the mapping.

    Return value: ctypes.c_void_p

    Parameters:
        runtime (`TiRuntime`):
        memory (`TiMemory`):
    """
    return _LIB.ti_map_memory(runtime, memory)


def ti_unmap_memory(
  runtime: TiRuntime,
  memory: TiMemory
) -> None:
    """
    Function `ti_unmap_memory` (1.4.0)
    
    Unmaps a device memory and makes any host-side changes about the memory visible to the device. You *must* ensure that there is no further access to the previously mapped host-addressable space.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        memory (`TiMemory`):
    """
    return _LIB.ti_unmap_memory(runtime, memory)


def ti_allocate_image(
  runtime: TiRuntime,
  allocate_info: ctypes.c_void_p, # const TiImageAllocateInfo*
) -> TiImage:
    """
    Function `ti_allocate_image` (1.4.0)
    
    Allocates a device image with provided parameters.

    Return value: TiImage

    Parameters:
        runtime (`TiRuntime`):
        allocate_info (`TiImageAllocateInfo`):
    """
    return _LIB.ti_allocate_image(runtime, allocate_info)


def ti_free_image(
  runtime: TiRuntime,
  image: TiImage
) -> None:
    """
    Function `ti_free_image` (1.4.0)
    
    Frees an image allocation.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        image (`TiImage`):
    """
    return _LIB.ti_free_image(runtime, image)


def ti_create_sampler(
  runtime: TiRuntime,
  create_info: ctypes.c_void_p, # const TiSamplerCreateInfo*
) -> TiSampler:
    """
    Function `ti_create_sampler`

    Return value: TiSampler

    Parameters:
        runtime (`TiRuntime`):
        create_info (`TiSamplerCreateInfo`):
    """
    return _LIB.ti_create_sampler(runtime, create_info)


def ti_destroy_sampler(
  runtime: TiRuntime,
  sampler: TiSampler
) -> None:
    """
    Function `ti_destroy_sampler`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        sampler (`TiSampler`):
    """
    return _LIB.ti_destroy_sampler(runtime, sampler)


def ti_copy_memory_device_to_device(
  runtime: TiRuntime,
  dst_memory: ctypes.c_void_p, # const TiMemorySlice*,
  src_memory: ctypes.c_void_p, # const TiMemorySlice*
) -> None:
    """
    Function `ti_copy_memory_device_to_device` (Device Command) (1.4.0)
    
    Copies the data in a contiguous subsection of the device memory to another subsection. The two subsections *must not* overlap.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        dst_memory (`TiMemorySlice`):
        src_memory (`TiMemorySlice`):
    """
    return _LIB.ti_copy_memory_device_to_device(runtime, dst_memory, src_memory)


def ti_copy_image_device_to_device(
  runtime: TiRuntime,
  dst_image: ctypes.c_void_p, # const TiImageSlice*,
  src_image: ctypes.c_void_p, # const TiImageSlice*
) -> None:
    """
    Function `ti_copy_image_device_to_device` (Device Command)
    
    Copies the image data in a contiguous subsection of the device image to another subsection. The two subsections *must not* overlap.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        dst_image (`TiImageSlice`):
        src_image (`TiImageSlice`):
    """
    return _LIB.ti_copy_image_device_to_device(runtime, dst_image, src_image)


def ti_track_image_ext(
  runtime: TiRuntime,
  image: TiImage,
  layout: TiImageLayout
) -> None:
    """
    Function `ti_track_image_ext` (1.4.0)
    
    Tracks the device image with the provided image layout. Because Taichi tracks image layouts internally, it is *only* useful to inform Taichi that the image is transitioned to a new layout by external procedures.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        image (`TiImage`):
        layout (`TiImageLayout`):
    """
    return _LIB.ti_track_image_ext(runtime, image, layout)


def ti_transition_image(
  runtime: TiRuntime,
  image: TiImage,
  layout: TiImageLayout
) -> None:
    """
    Function `ti_transition_image` (Device Command) (1.4.0)
    
    Transitions the image to the provided image layout. Because Taichi tracks image layouts internally, it is *only* useful to enforce an image layout for external procedures to use.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        image (`TiImage`):
        layout (`TiImageLayout`):
    """
    return _LIB.ti_transition_image(runtime, image, layout)


def ti_launch_kernel(
  runtime: TiRuntime,
  kernel: TiKernel,
  arg_count: ctypes.c_uint32,
  args: ctypes.c_void_p, # const TiArgument*
) -> None:
    """
    Function `ti_launch_kernel` (Device Command) (1.4.0)
    
    Launches a Taichi kernel with the provided arguments. The arguments *must* have the same count and types in the same order as in the source code.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        kernel (`TiKernel`):
        arg_count (`ctypes.c_uint32`):
        args (`TiArgument`):
    """
    return _LIB.ti_launch_kernel(runtime, kernel, arg_count, args)


def ti_launch_compute_graph(
  runtime: TiRuntime,
  compute_graph: TiComputeGraph,
  arg_count: ctypes.c_uint32,
  args: ctypes.c_void_p, # const TiNamedArgument*
) -> None:
    """
    Function `ti_launch_compute_graph` (Device Command) (1.4.0)
    
    Launches a Taichi compute graph with provided named arguments. The named arguments *must* have the same count, names, and types as in the source code.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        compute_graph (`TiComputeGraph`):
        arg_count (`ctypes.c_uint32`):
        args (`TiNamedArgument`):
    """
    return _LIB.ti_launch_compute_graph(runtime, compute_graph, arg_count, args)


def ti_flush(
  runtime: TiRuntime
) -> None:
    """
    Function `ti_flush` (1.4.0)
    
    Submits all previously invoked device commands to the offload device for execution.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
    """
    return _LIB.ti_flush(runtime)


def ti_wait(
  runtime: TiRuntime
) -> None:
    """
    Function `ti_wait` (1.4.0)
    
    Waits until all previously invoked device commands are executed. Any invoked command that has not been submitted is submitted first.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
    """
    return _LIB.ti_wait(runtime)


def ti_load_aot_module(
  runtime: TiRuntime,
  module_path: ctypes.c_void_p
) -> TiAotModule:
    """
    Function `ti_load_aot_module` (1.4.0)
    
    Loads a pre-compiled AOT module from the file system.
    Returns [`TI_NULL_HANDLE`](#definition-ti_null_handle) if the runtime fails to load the AOT module from the specified path.

    Return value: TiAotModule

    Parameters:
        runtime (`TiRuntime`):
        module_path (`ctypes.c_void_p`):
    """
    return _LIB.ti_load_aot_module(runtime, module_path)


def ti_create_aot_module(
  runtime: TiRuntime,
  tcm: ctypes.c_void_p,
  size: ctypes.c_uint64
) -> TiAotModule:
    """
    Function `ti_create_aot_module` (1.4.0)
    
    Creates a pre-compiled AOT module from TCM data.
    Returns [`TI_NULL_HANDLE`](#definition-ti_null_handle) if the runtime fails to create the AOT module from TCM data.

    Return value: TiAotModule

    Parameters:
        runtime (`TiRuntime`):
        tcm (`ctypes.c_void_p`):
        size (`ctypes.c_uint64`):
    """
    return _LIB.ti_create_aot_module(runtime, tcm, size)


def ti_destroy_aot_module(
  aot_module: TiAotModule
) -> None:
    """
    Function `ti_destroy_aot_module` (1.4.0)
    
    Destroys a loaded AOT module and releases all related resources.

    Return value: None

    Parameters:
        aot_module (`TiAotModule`):
    """
    return _LIB.ti_destroy_aot_module(aot_module)


def ti_get_aot_module_kernel(
  aot_module: TiAotModule,
  name: ctypes.c_void_p
) -> TiKernel:
    """
    Function `ti_get_aot_module_kernel` (1.4.0)
    
    Retrieves a pre-compiled Taichi kernel from the AOT module.
    Returns [`TI_NULL_HANDLE`](#definition-ti_null_handle) if the module does not have a kernel of the specified name.

    Return value: TiKernel

    Parameters:
        aot_module (`TiAotModule`):
        name (`ctypes.c_void_p`):
    """
    return _LIB.ti_get_aot_module_kernel(aot_module, name)


def ti_get_aot_module_compute_graph(
  aot_module: TiAotModule,
  name: ctypes.c_void_p
) -> TiComputeGraph:
    """
    Function `ti_get_aot_module_compute_graph` (1.4.0)
    
    Retrieves a pre-compiled compute graph from the AOT module.
    Returns [`TI_NULL_HANDLE`](#definition-ti_null_handle) if the module does not have a compute graph of the specified name.

    Return value: TiComputeGraph

    Parameters:
        aot_module (`TiAotModule`):
        name (`ctypes.c_void_p`):
    """
    return _LIB.ti_get_aot_module_compute_graph(aot_module, name)
