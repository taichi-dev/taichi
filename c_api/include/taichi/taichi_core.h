#pragma once

#include "taichi/taichi_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t TiBool;
#define TI_TRUE 1u
#define TI_FALSE 0u

typedef uint32_t TiFlags;

typedef void *TiDispatchableHandle;
typedef size_t TiNonDispatchableHandle;
#define TI_NULL_HANDLE 0

typedef TiDispatchableHandle TiRuntime;
typedef TiDispatchableHandle TiAotModule;

typedef TiNonDispatchableHandle TiMemory;
typedef TiNonDispatchableHandle TiKernel;
typedef TiNonDispatchableHandle TiComputeGraph;

typedef enum TiArch {
  // TI_ARCH_ANY = 0,  // TODO: (penguinliong) Do we need this?
  TI_ARCH_VULKAN = 1,
} TiArch;
TI_DLL_EXPORT TiRuntime TI_API_CALL ti_create_runtime(TiArch arch);
TI_DLL_EXPORT void TI_API_CALL ti_destroy_runtime(TiRuntime runtime);

typedef enum TiAllocationUsageFlagBits {
  TI_MEMORY_USAGE_STORAGE_BIT = 1,
  TI_MEMORY_USAGE_UNIFORM_BIT = 2,
  TI_MEMORY_USAGE_VERTEX_BIT = 4,
  TI_MEMORY_USAGE_INDEX_BIT = 8,
} TiMemoryUsageFlagBits;
typedef TiFlags TiMemoryUsageFlags;
typedef struct {
  uint64_t size;
  TiBool host_write;
  TiBool host_read;
  TiBool export_sharing;
  TiMemoryUsageFlags usage;
} TiMemoryAllocateInfo;
TI_DLL_EXPORT TiMemory TI_API_CALL
ti_allocate_memory(TiRuntime runtime,
                   const TiMemoryAllocateInfo *allocate_info);
TI_DLL_EXPORT void TI_API_CALL ti_free_memory(TiRuntime runtime,
                                              TiMemory memory);
TI_DLL_EXPORT void *TI_API_CALL ti_map_memory(TiRuntime runtime,
                                              TiMemory memory);
TI_DLL_EXPORT void TI_API_CALL ti_unmap_memory(TiRuntime runtime,
                                               TiMemory memory);

typedef struct TiNdShape {
  uint32_t dim_count;
  uint32_t dims[16];  // TODO: (penguinliong) give this constant a name?
} TiNdShape;
typedef struct TiNdArray {
  TiMemory memory;
  TiNdShape shape;
  TiNdShape elem_shape;
} TiNdArray;
enum TiArgumentType {
  TI_ARGUMENT_TYPE_I32,
  TI_ARGUMENT_TYPE_F32,
  TI_ARGUMENT_TYPE_NDARRAY,
};
union TiArgumentValue {
  int32_t i32;
  float f32;
  TiNdArray ndarray;
};
struct TiArgument {
  TiArgumentType type;
  TiArgumentValue value;
};
struct TiNamedArgument {
  const char *name;
  TiArgument arg;
};
TI_DLL_EXPORT void TI_API_CALL ti_launch_kernel(TiRuntime runtime,
                                                TiKernel kernel,
                                                uint32_t arg_count,
                                                const TiArgument *args);
TI_DLL_EXPORT void TI_API_CALL
ti_launch_compute_graph(TiRuntime runtime,
                        TiComputeGraph compute_graph,
                        uint32_t arg_count,
                        const TiNamedArgument *named_args);
TI_DLL_EXPORT void TI_API_CALL ti_submit(TiRuntime runtime);
TI_DLL_EXPORT void TI_API_CALL ti_wait(TiRuntime runtime);

TI_DLL_EXPORT TiAotModule TI_API_CALL
ti_load_aot_module(TiRuntime runtime, const char *module_path);
TI_DLL_EXPORT void TI_API_CALL ti_destroy_aot_module(TiAotModule mod);
TI_DLL_EXPORT TiKernel TI_API_CALL ti_get_aot_module_kernel(TiAotModule mod,
                                                            const char *name);
TI_DLL_EXPORT TiComputeGraph TI_API_CALL
ti_get_aot_module_compute_graph(TiAotModule mod, const char *name);

#ifdef __cplusplus
}  // extern "C"
#endif
