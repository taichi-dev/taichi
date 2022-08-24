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

// handle.texture
typedef struct TiTexture_t *TiTexture;

// handle.kernel
typedef struct TiKernel_t *TiKernel;

// handle.compute_graph
typedef struct TiComputeGraph_t *TiComputeGraph;

// enumeration.error
typedef enum TiError {
  TI_ERROR_INCOMPLETE = 1,
  TI_ERROR_SUCCESS = 0,
  TI_ERROR_NOT_SUPPORTED = -1,
  TI_ERROR_CORRUPTED_DATA = -2,
  TI_ERROR_NAME_NOT_FOUND = -3,
  TI_ERROR_INVALID_ARGUMENT = -4,
  TI_ERROR_ARGUMENT_NULL = -5,
  TI_ERROR_ARGUMENT_OUT_OF_RANGE = -6,
  TI_ERROR_ARGUMENT_NOT_FOUND = -7,
  TI_ERROR_INVALID_INTEROP = -8,
  TI_ERROR_MAX_ENUM = 0xffffffff,
} TiError;

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
  TI_ARGUMENT_TYPE_TEXTURE = 3,
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

// bit_field.texture_usage
typedef enum TiTextureUsageFlagBits {
  TI_TEXTURE_USAGE_STORAGE_BIT = 1 << 0,
  TI_TEXTURE_USAGE_SAMPLED_BIT = 1 << 1,
  TI_TEXTURE_USAGE_ATTACHMENT_BIT = 1 << 2,
} TiTextureUsageFlagBits;
typedef TiFlags TiTextureUsageFlags;

// enumeration.texture_dimension
typedef enum TiTextureDimension {
  TI_TEXTURE_DIMENSION_1D = 0,
  TI_TEXTURE_DIMENSION_2D = 1,
  TI_TEXTURE_DIMENSION_3D = 2,
  TI_TEXTURE_DIMENSION_1D_ARRAY = 3,
  TI_TEXTURE_DIMENSION_2D_ARRAY = 4,
  TI_TEXTURE_DIMENSION_CUBE = 5,
  TI_TEXTURE_DIMENSION_MAX_ENUM = 0xffffffff,
} TiTextureDimension;

// enumeration.texture_layout
typedef enum TiTextureLayout {
  TI_TEXTURE_LAYOUT_UNDEFINED = 0,
  TI_TEXTURE_LAYOUT_SHADER_READ = 1,
  TI_TEXTURE_LAYOUT_SHADER_WRITE = 2,
  TI_TEXTURE_LAYOUT_SHADER_READ_WRITE = 3,
  TI_TEXTURE_LAYOUT_COLOR_ATTACHMENT = 4,
  TI_TEXTURE_LAYOUT_COLOR_ATTACHMENT_READ = 5,
  TI_TEXTURE_LAYOUT_DEPTH_ATTACHMENT = 6,
  TI_TEXTURE_LAYOUT_DEPTH_ATTACHMENT_READ = 7,
  TI_TEXTURE_LAYOUT_TRANSFER_DST = 8,
  TI_TEXTURE_LAYOUT_TRANSFER_SRC = 9,
  TI_TEXTURE_LAYOUT_PRESENT_SRC = 10,
  TI_TEXTURE_LAYOUT_MAX_ENUM = 0xffffffff,
} TiTextureLayout;

// enumeration.texture_format
typedef enum TiTextureFormat {
  TI_TEXTURE_FORMAT_UNKNOWN = 0,
  TI_TEXTURE_FORMAT_R8 = 1,
  TI_TEXTURE_FORMAT_RG8 = 2,
  TI_TEXTURE_FORMAT_RGBA8 = 3,
  TI_TEXTURE_FORMAT_RGBA8SRGB = 4,
  TI_TEXTURE_FORMAT_BGRA8 = 5,
  TI_TEXTURE_FORMAT_BGRA8SRGB = 6,
  TI_TEXTURE_FORMAT_R8U = 7,
  TI_TEXTURE_FORMAT_RG8U = 8,
  TI_TEXTURE_FORMAT_RGBA8U = 9,
  TI_TEXTURE_FORMAT_R8I = 10,
  TI_TEXTURE_FORMAT_RG8I = 11,
  TI_TEXTURE_FORMAT_RGBA8I = 12,
  TI_TEXTURE_FORMAT_R16 = 13,
  TI_TEXTURE_FORMAT_RG16 = 14,
  TI_TEXTURE_FORMAT_RGB16 = 15,
  TI_TEXTURE_FORMAT_RGBA16 = 16,
  TI_TEXTURE_FORMAT_R16U = 17,
  TI_TEXTURE_FORMAT_RG16U = 18,
  TI_TEXTURE_FORMAT_RGB16U = 19,
  TI_TEXTURE_FORMAT_RGBA16U = 20,
  TI_TEXTURE_FORMAT_R16I = 21,
  TI_TEXTURE_FORMAT_RG16I = 22,
  TI_TEXTURE_FORMAT_RGB16I = 23,
  TI_TEXTURE_FORMAT_RGBA16I = 24,
  TI_TEXTURE_FORMAT_R16F = 25,
  TI_TEXTURE_FORMAT_RG16F = 26,
  TI_TEXTURE_FORMAT_RGB16F = 27,
  TI_TEXTURE_FORMAT_RGBA16F = 28,
  TI_TEXTURE_FORMAT_R32U = 29,
  TI_TEXTURE_FORMAT_RG32U = 30,
  TI_TEXTURE_FORMAT_RGB32U = 31,
  TI_TEXTURE_FORMAT_RGBA32U = 32,
  TI_TEXTURE_FORMAT_R32I = 33,
  TI_TEXTURE_FORMAT_RG32I = 34,
  TI_TEXTURE_FORMAT_RGB32I = 35,
  TI_TEXTURE_FORMAT_RGBA32I = 36,
  TI_TEXTURE_FORMAT_R32F = 37,
  TI_TEXTURE_FORMAT_RG32F = 38,
  TI_TEXTURE_FORMAT_RGB32F = 39,
  TI_TEXTURE_FORMAT_RGBA32F = 40,
  TI_TEXTURE_FORMAT_DEPTH16 = 41,
  TI_TEXTURE_FORMAT_DEPTH24STENCIL8 = 42,
  TI_TEXTURE_FORMAT_DEPTH32F = 43,
  TI_TEXTURE_FORMAT_MAX_ENUM = 0xffffffff,
} TiTextureFormat;

// structure.texture_offset
typedef struct TiTextureOffset {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t array_layer_offset;
} TiTextureOffset;

// structure.texture_extent
typedef struct TiTextureExtent {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t array_layer_count;
} TiTextureExtent;

// structure.texture_allocate_info
typedef struct TiTextureAllocateInfo {
  TiTextureDimension dimension;
  TiTextureExtent extent;
  uint32_t mip_level_count;
  TiTextureFormat format;
  TiTextureUsageFlagBits usage;
} TiTextureAllocateInfo;

// structure.texture_slice
typedef struct TiTextureSlice {
  TiTexture texture;
  TiTextureOffset offset;
  TiTextureExtent extent;
  uint32_t mip_level;
} TiTextureSlice;

// union.argument_value
typedef union TiArgumentValue {
  int32_t i32;
  float f32;
  TiNdArray ndarray;
  TiTexture texture;
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

// function.get_last_error
TI_DLL_EXPORT TiError TI_API_CALL ti_get_last_error(uint64_t message_size,
                                                    char *message);

// function.set_last_error
TI_DLL_EXPORT void TI_API_CALL ti_set_last_error(TiError error,
                                                 const char *message);

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

// function.allocate_texture
TI_DLL_EXPORT TiTexture TI_API_CALL
ti_allocate_texture(TiRuntime runtime,
                    const TiTextureAllocateInfo *allocate_info);

// function.free_texture
TI_DLL_EXPORT void TI_API_CALL ti_free_texture(TiRuntime runtime,
                                               TiTexture texture);

// function.create_event
TI_DLL_EXPORT TiEvent TI_API_CALL ti_create_event(TiRuntime runtime);

// function.destroy_event
TI_DLL_EXPORT void TI_API_CALL ti_destroy_event(TiEvent event);

// function.copy_memory_device_to_device
TI_DLL_EXPORT void TI_API_CALL
ti_copy_memory_device_to_device(TiRuntime runtime,
                                const TiMemorySlice *dst_memory,
                                const TiMemorySlice *src_memory);

// function.copy_texture_device_to_device
TI_DLL_EXPORT void TI_API_CALL
ti_copy_texture_device_to_device(TiRuntime runtime,
                                 const TiTextureSlice *dst_texture,
                                 const TiTextureSlice *src_texture);

// function.transition_texture
TI_DLL_EXPORT void TI_API_CALL ti_transition_texture(TiRuntime runtime,
                                                     TiTexture texture,
                                                     TiTextureLayout layout);

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

// function.wait_event
TI_DLL_EXPORT void TI_API_CALL ti_wait_event(TiRuntime runtime, TiEvent event);

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
