#pragma once
#include <taichi/taichi_platform.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// alias.bool
typedef uint32_t TiBool;

// definition.false
#define TI_FALSE 0

// definition.true
#define TI_TRUE 1

// alias.flags
typedef uint32_t TiFlags;

// definition.null_handle
#define TI_NULL_HANDLE 0

// handle.runtime
typedef struct TiRuntime_t *TiRuntime;

// handle.aot_module
typedef struct TiAotModule_t *TiAotModule;

// handle.event
typedef struct TiEvent_t *TiEvent;

// handle.memory
typedef struct TiMemory_t *TiMemory;

// handle.kernel
typedef struct TiKernel_t *TiKernel;

// handle.compute_graph
typedef struct TiComputeGraph_t *TiComputeGraph;

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

// enumeration.argument_type
typedef enum TiArgumentType {
  TI_ARGUMENT_TYPE_I32 = 0,
  TI_ARGUMENT_TYPE_F32 = 1,
  TI_ARGUMENT_TYPE_NDARRAY = 2,
  TI_ARGUMENT_TYPE_MAX_ENUM = 0xffffffff,
} TiArgumentType;

// bit_field.memory_usage
typedef enum TiMemoryUsageFlagBits {
  TI_MEMORY_USAGE_STORAGE_BIT = 1 << 0,
  TI_MEMORY_USAGE_UNIFORM_BIT = 1 << 1,
  TI_MEMORY_USAGE_VERTEX_BIT = 1 << 2,
  TI_MEMORY_USAGE_INDEX_BIT = 1 << 3,
} TiMemoryUsageFlagBits;
typedef TiFlags TiMemoryUsageFlags;

// structure.memory_allocate_info
typedef struct TiMemoryAllocateInfo {
  uint64_t size;
  TiBool host_write;
  TiBool host_read;
  TiBool export_sharing;
  TiMemoryUsageFlagBits usage;
} TiMemoryAllocateInfo;

// structure.memory_slice
typedef struct TiMemorySlice {
  TiMemory memory;
  uint64_t offset;
  uint64_t size;
} TiMemorySlice;

// structure.nd_shape
typedef struct TiNdShape {
  uint32_t dim_count;
  uint32_t dims[16];
} TiNdShape;

// structure.nd_array
typedef struct TiNdArray {
  TiMemory memory;
  TiNdShape shape;
  TiNdShape elem_shape;
  TiDataType elem_type;
} TiNdArray;

// union.argument_value
typedef union TiArgumentValue {
  int32_t i32;
  float f32;
  TiNdArray ndarray;
} TiArgumentValue;

// structure.argument
typedef struct TiArgument {
  TiArgumentType type;
  TiArgumentValue value;
} TiArgument;

// structure.named_argument
typedef struct TiNamedArgument {
  const char *name;
  TiArgument argument;
} TiNamedArgument;

// function.create_runtime
TI_DLL_EXPORT TiRuntime TI_API_CALL ti_create_runtime(TiArch arch);

// function.destroy_runtime
TI_DLL_EXPORT void TI_API_CALL ti_destroy_runtime(TiRuntime runtime);

// function.allocate_memory
TI_DLL_EXPORT TiMemory TI_API_CALL
ti_allocate_memory(TiRuntime runtime,
                   const TiMemoryAllocateInfo *allocate_info);

// function.free_memory
TI_DLL_EXPORT void TI_API_CALL ti_free_memory(TiRuntime runtime,
                                              TiMemory memory);

// function.map_memory
TI_DLL_EXPORT void *TI_API_CALL ti_map_memory(TiRuntime runtime,
                                              TiMemory memory);

// function.unmap_memory
TI_DLL_EXPORT void TI_API_CALL ti_unmap_memory(TiRuntime runtime,
                                               TiMemory memory);

// function.create_event
TI_DLL_EXPORT TiEvent TI_API_CALL ti_create_event(TiRuntime runtime);

// function.destroy_event
TI_DLL_EXPORT void TI_API_CALL ti_destroy_event(TiEvent event);

// function.copy_memory_device_to_device
TI_DLL_EXPORT void TI_API_CALL
ti_copy_memory_device_to_device(TiRuntime runtime,
                                const TiMemorySlice *dst_memory,
                                const TiMemorySlice *src_memory);

// function.launch_kernel
TI_DLL_EXPORT void TI_API_CALL ti_launch_kernel(TiRuntime runtime,
                                                TiKernel kernel,
                                                uint32_t arg_count,
                                                const TiArgument *args);

// function.launch_compute_graph
TI_DLL_EXPORT void TI_API_CALL
ti_launch_compute_graph(TiRuntime runtime,
                        TiComputeGraph compute_graph,
                        uint32_t arg_count,
                        const TiNamedArgument *args);

// function.signal_event
TI_DLL_EXPORT void TI_API_CALL ti_signal_event(TiRuntime runtime,
                                               TiEvent event);

// function.reset_event
TI_DLL_EXPORT void TI_API_CALL ti_reset_event(TiRuntime runtime, TiEvent event);

// function.submit
TI_DLL_EXPORT void TI_API_CALL ti_submit(TiRuntime runtime);

// function.wait
TI_DLL_EXPORT void TI_API_CALL ti_wait(TiRuntime runtime);

// function.load_aot_module
TI_DLL_EXPORT TiAotModule TI_API_CALL
ti_load_aot_module(TiRuntime runtime, const char *module_path);

// function.destroy_aot_module
TI_DLL_EXPORT void TI_API_CALL ti_destroy_aot_module(TiAotModule aot_module);

// function.get_aot_module_kernel
TI_DLL_EXPORT TiKernel TI_API_CALL
ti_get_aot_module_kernel(TiAotModule aot_module, const char *name);

// function.get_aot_module_compute_graph
TI_DLL_EXPORT TiComputeGraph TI_API_CALL
ti_get_aot_module_compute_graph(TiAotModule aot_module, const char *name);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
