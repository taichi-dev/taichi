// clang-format off
#pragma once

#if defined(TI_EXPORTS_TO_PY)
#define _Atomic(x) x  // To make pycparser happy
#endif // defined(TI_EXPORTS_TO_PY)

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#if defined(TI_EXPORTS_TO_PY)
#define TI_DLL_EXPORT
#define TI_API_CALL
#else
// https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined _WIN64 || defined __CYGWIN__
#ifdef __GNUC__
#define TI_DLL_EXPORT __attribute__((dllexport))
#define TI_API_CALL
#else
#define TI_DLL_EXPORT __declspec(dllexport)
#define TI_API_CALL __stdcall
#endif  //  __GNUC__
#else
#define TI_DLL_EXPORT __attribute__((visibility("default")))
#define TI_API_CALL
#endif  // defined _WIN32 || defined _WIN64 || defined __CYGWIN__
#endif  // defined(TI_EXPORTS_TO_PY)

// Windows
#if defined(_WIN64)
#define TI_PLATFORM_WINDOWS
#endif

#if defined(_WIN32) && !defined(_WIN64)
static_assert(false, "32-bit Windows systems are not supported")
#endif

// Linux
#if defined(__linux__)
#if defined(ANDROID)
#define TI_PLATFORM_ANDROID
#else
#define TI_PLATFORM_LINUX
#endif
#endif

// OSX
#if defined(__APPLE__)
#define TI_PLATFORM_OSX
#endif

#if (defined(TI_PLATFORM_LINUX) || defined(TI_PLATFORM_OSX) || \
     defined(__unix__))
#define TI_PLATFORM_UNIX
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Tie, tie, TIE: taichi exports

// Func signature:
//     Rules:
//         1. All functions are prefixed with "tie_", which means "taichi exports".
//         2. The name of general parameters can be arbitrary, but cannot be "self", "ret_" or "ap_" prefixed.
//             2.1 The parameter named "self" is the object itself.
//             2.2 The parameter named "ret_" prefixed is the return value (output parameter).
//             2.3 The parameter named "ap_" prefixed is the array type parameter, and the parameter specifying the size of the array must follow it.
//         3. Return type is always int, which is the error code (TieError).
//         4. Return value (except error code) is always the last parameters. And the parameter name always starts with "ret_".
//     1. Create Object: int TI_API_CALL tie_<ClassName>_create(<Params...>, Tie<ClassName>Handle *ret_handle);
//     2. Destroy Object: int TI_API_CALL tie_<ClassName>_destroy(Tie<ClassName>Handle self);
//     3. Member Function: int TI_API_CALL tie_<ClassName>_<MemFuncName>(Tie<ClassName>Handle self, <Params...>, <RetType> *ret_<RetName>);
//     4. Static Function: int TI_API_CALL tie_<ClassName>_<StaticFuncName>(<Params...>, <RetType> *ret_<RetName>);
//     5. Global Function: int TI_API_CALL tie_G_<GlobalFuncName>(<Params...>, <RetType> *ret_<RetName>);

typedef enum TieError {
  // Success (always 0)
  TIE_ERROR_SUCCESS = 0,

  // Invalid argument
  TIE_ERROR_INVALID_ARGUMENT = -101,
  TIE_ERROR_INVALID_RETURN_ARG = -102,
  TIE_ERROR_INVALID_HANDLE = -103,
  TIE_ERROR_INVALID_INDEX = -104,

  // CXX exceptions
  TIE_ERROR_TAICHI_TYPE_ERROR = -201,       // taichi::lang::TaichiTypeError
  TIE_ERROR_TAICHI_SYNTAX_ERROR = -202,     // taichi::lang::TaichiSyntaxError
  TIE_ERROR_TAICHI_INDEX_ERROR = -203,      // taichi::lang::TaichiIndexError
  TIE_ERROR_TAICHI_RUNTIME_ERROR = -204,    // taichi::lang::TaichiRuntimeError
  TIE_ERROR_TAICHI_ASSERTION_ERROR = -205,  // taichi::lang::TaichiAssertionError
  TIE_ERROR_OUT_OF_MEMORY = -298,           // std::bad_alloc
  TIE_ERROR_UNKNOWN_CXX_EXCEPTION = -299,   // std::exception

  // Callback failed
  TIE_ERROR_CALLBACK_FAILED = -301,

  // Unknown error
  TIE_ERROR_UNKNOWN = -0x7FFFFFFF // INT_MIN
} TieError;

typedef enum TieArch {
  // CPU archs
  TIE_ARCH_X64 = 0,    // a.k.a. AMD64/x86_64
  TIE_ARCH_ARM64 = 1,  // a.k.a. Aarch64
  TIE_ARCH_JS = 2,     // Javascript

  // GPU archs
  TIE_ARCH_CUDA = 3,     // NVIDIA CUDA
  TIE_ARCH_METAL = 4,    // Apple Metal
  TIE_ARCH_OPENGL = 5,   // OpenGL Compute Shaders
  TIE_ARCH_DX11 = 6,     // Microsoft DirectX 11
  TIE_ARCH_DX12 = 7,     // Microsoft DirectX 12
  TIE_ARCH_OPENCL = 8,   // OpenCL, N/A
  TIE_ARCH_AMDGPU = 9,   // AMD GPU
  TIE_ARCH_VULKAN = 10,  // Vulkan
  TIE_ARCH_GLES = 11,    // OpenGL ES

  // Unknown arch
  TIE_ARCH_UNKNOWN = 0x7FFFFFFF // INT_MAX
} TieArch;

typedef enum TieExtension {
  TIE_EXTENSION_SPARSE = 0,       // Sparse data structures
  TIE_EXTENSION_QUANT = 1,        // Quantization
  TIE_EXTENSION_MESH = 2,         // MeshTaichi
  TIE_EXTENSION_QUANT_BASIC = 3,  // Basic operations in quantization
  TIE_EXTENSION_DATA64 = 4,       // Metal doesn't support 64-bit data buffers yet...
  TIE_EXTENSION_ADSTACK = 5,      // For keeping the history of mutable local variables
  TIE_EXTENSION_BLS = 6,          // Block-local storage
  TIE_EXTENSION_ASSERTION = 7,    // Run-time asserts in Taichi kernels
  TIE_EXTENSION_EXTFUNC = 8,      // Invoke external functions or backend source

  // Unknown extension
  TIE_EXTENSION_UNKNOWN = 0x7FFFFFFF  // INT_MAX
} TieExtension;


typedef int (*TieCallback)(void);  // Return 0 if success, otherwise -1.

typedef void *TieHandle;
typedef TieHandle TieStringHandle;                     // std::string
typedef TieHandle TieCompileConfigHandle;              // class CompileConfig
typedef TieHandle TieKernelHandle;                     // class Kernel
typedef TieHandle TieFunctionHandle;                   // class Function
typedef TieHandle TieNdarrayHandle;                    // class NDArray
typedef TieHandle TieTextureHandle;                    // class Texture
typedef TieHandle TieLaunchContextBuilderHandle;       // class LaunchContextBuilder
typedef TieHandle TieDataTypeHandle;                   // class DataType
typedef TieHandle TieASTBuilderHandle;                 // class ASTBuilder
typedef TieHandle TieSNodeHandle;                      // class SNode
typedef TieHandle TieProgramHandle;                    // class Program
typedef TieHandle TieAotModuleBuilderHandle;           // class AotModuleBuilder
typedef TieHandle TieSparseMatrixHandle;               // class SparseMatrix
typedef TieHandle TieCompiledKernelDataHandle;         // class CompiledKernelData
typedef TieHandle TieKernelProfileTracedRecordHandle;  // struct KernelProfileTracedRecord

typedef void *TieRef;
typedef TieRef TieStringRef;                     // std::string
typedef TieRef TieCompileConfigRef;              // class CompileConfig
typedef TieRef TieKernelRef;                     // class Kernel
typedef TieRef TieFunctionRef;                   // class Function
typedef TieRef TieNdarrayRef;                    // class NDArray
typedef TieRef TieTextureRef;                    // class Texture
typedef TieRef TieLaunchContextBuilderRef;       // class LaunchContextBuilder
typedef TieRef TieDataTypeRef;                   // class DataType
typedef TieRef TieASTBuilderRef;                 // class ASTBuilder
typedef TieRef TieSNodeRef;                      // class SNode
typedef TieRef TieProgramRef;                    // class Program
typedef TieRef TieAotModuleBuilderRef;           // class AotModuleBuilder
typedef TieRef TieSparseMatrixRef;               // class SparseMatrix
typedef TieRef TieCompiledKernelDataRef;         // class CompiledKernelData
typedef TieRef TieKernelProfileTracedRecordRef;  // struct KernelProfileTracedRecord

// std::string
TI_DLL_EXPORT int TI_API_CALL tie_String_create(const char *str, TieStringHandle *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_String_destroy(TieStringHandle self);

TI_DLL_EXPORT int TI_API_CALL tie_String_c_str(TieStringHandle self, const char **ret_c_str);

TI_DLL_EXPORT int TI_API_CALL tie_String_size(TieStringHandle self, size_t *ret_size);

// Error processing
TI_DLL_EXPORT int TI_API_CALL tie_G_set_last_error(int error, const char *msg);

TI_DLL_EXPORT int TI_API_CALL tie_G_get_last_error(int *ret_error, const char **ret_msg);

// Arch handling
TI_DLL_EXPORT int TI_API_CALL tie_G_arch_name(int arch, const char **ret_name);

TI_DLL_EXPORT int TI_API_CALL tie_G_arch_from_name(const char *name, int *ret_arch);

TI_DLL_EXPORT int TI_API_CALL tie_G_host_arch(int *ret_arch);

TI_DLL_EXPORT int TI_API_CALL tie_G_is_extension_supported(int arch, int extension, bool *ret_supported);

// default_compile_config handling
TI_DLL_EXPORT int TI_API_CALL tie_G_default_compile_config(TieCompileConfigRef *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_G_reset_default_compile_config();

// class CompileConfig
TI_DLL_EXPORT int TI_API_CALL tie_CompileConfig_create(TieCompileConfigHandle *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_CompileConfig_destroy(TieCompileConfigRef self);

#define TIE_DECL_COMPILE_CONFIG_GET_SET_ATTR(TaichiStruct, attr_name, attr_type, get_set_type) \
  TI_DLL_EXPORT int TI_API_CALL tie_CompileConfig_get_##attr_name(TieCompileConfigRef self, get_set_type *ret_##attr_name); \
  TI_DLL_EXPORT int TI_API_CALL tie_CompileConfig_set_##attr_name(TieCompileConfigRef self, get_set_type attr_name);
#define TIE_PER_COMPILE_CONFIG_ATTR TIE_DECL_COMPILE_CONFIG_GET_SET_ATTR
#include "inc/compile_config.inc.h"
#undef TIE_PER_COMPILE_CONFIG_ATTR
#undef TIE_DECL_COMPILE_CONFIG_GET_SET_ATTR

// class Kernel
TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_scalar_param(TieKernelRef self, TieDataTypeRef dt, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_arr_param(TieKernelRef self, TieDataTypeRef dt, int total_dim, int *ap_element_shape, size_t element_shape_dim, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_ndarray_param(TieKernelRef self, TieDataTypeRef dt, int ndim, const char *name, int needs_grad, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_texture_param(TieKernelRef self, int total_dim, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_pointer_param(TieKernelRef self, TieDataTypeRef dt, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_rw_texture_param(TieKernelRef self, int total_dim, int format, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_ret(TieKernelRef self, TieDataTypeRef dt, int *ret_ret_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_finalize_rets(TieKernelRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_finalize_params(TieKernelRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_ast_builder(TieKernelRef self, TieASTBuilderRef *ret_ast_builder);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_no_activate(TieKernelRef self, TieSNodeRef snode);

// class Function
TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_scalar_param(TieFunctionRef self, TieDataTypeRef dt, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_arr_param(TieFunctionRef self, TieDataTypeRef dt, int total_dim, int *ap_element_shape, size_t element_shape_dim, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_ndarray_param(TieFunctionRef self, TieDataTypeRef dt, int ndim, const char *name, int needs_grad, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_texture_param(TieFunctionRef self, int total_dim, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_pointer_param(TieFunctionRef self, TieDataTypeRef dt, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_rw_texture_param(TieFunctionRef self, int total_dim, int format, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_set_function_body(TieFunctionRef self, TieCallback func);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_ret(TieFunctionRef self, TieDataTypeRef dt, int *ret_ret_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_finalize_rets(TieFunctionRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Function_finalize_params(TieFunctionRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Function_ast_builder(TieFunctionRef self, TieASTBuilderRef *ret_ast_builder);

// class LaunchContextBuilder
TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_create(TieKernelRef kernel_handle, TieLaunchContextBuilderHandle *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_destroy(TieLaunchContextBuilderRef self);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_int(TieLaunchContextBuilderRef self, int arg_id, int64_t i64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_uint(TieLaunchContextBuilderRef self, int arg_id, uint64_t u64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_float(TieLaunchContextBuilderRef self, int arg_id, double d);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_struct_arg_int(TieLaunchContextBuilderRef self, int *ap_arg_indices, size_t arg_indices_dim, int64_t i64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_struct_arg_uint(TieLaunchContextBuilderRef self, int *ap_arg_indices, size_t arg_indices_dim, uint64_t u64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_struct_arg_float(TieLaunchContextBuilderRef self, int *ap_arg_indices, size_t arg_indices_dim, double d);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_external_array_with_shape(TieLaunchContextBuilderRef self, int arg_id, uintptr_t ptr, uint64_t size, int64_t *ap_shape, size_t shape_dim, uintptr_t grad_ptr);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_ndarray(TieLaunchContextBuilderRef self, int arg_id, TieNdarrayRef arr);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_ndarray_with_grad(TieLaunchContextBuilderRef self, int arg_id, TieNdarrayRef arr, TieNdarrayRef arr_grad);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_texture(TieLaunchContextBuilderRef self, int arg_id, TieTextureRef tex);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_rw_texture(TieLaunchContextBuilderRef self, int arg_id, TieTextureRef tex);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_get_struct_ret_int(TieLaunchContextBuilderRef self, int *ap_index, size_t index_dim, int64_t *ret_i64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_get_struct_ret_uint(TieLaunchContextBuilderRef self, int *ap_index, size_t index_dim, uint64_t *ret_u64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_get_struct_ret_float(TieLaunchContextBuilderRef self, int *ap_index, size_t index_dim, double *ret_d);

// class Program

TI_DLL_EXPORT int TI_API_CALL tie_Program_create(TieProgramHandle *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_Program_destroy(TieProgramRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Program_finalize(TieProgramRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Program_synchronize(TieProgramRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Program_config(TieProgramRef self, TieCompileConfigRef *ret_config);

TI_DLL_EXPORT int TI_API_CALL tie_Program_sync_kernel_profiler(TieProgramRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Program_update_kernel_profiler(TieProgramRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Program_clear_kernel_profiler(TieProgramRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Program_query_kernel_profile_info(TieProgramRef self, const char *name, int *ret_counter, double *ret_min, double *ret_max, double *ret_avg);

TI_DLL_EXPORT int TI_API_CALL tie_Program_get_num_kernel_profiler_records(TieProgramRef self, size_t *ret_size);

TI_DLL_EXPORT int TI_API_CALL tie_Program_get_kernel_profiler_record(TieProgramRef self, size_t index, TieKernelProfileTracedRecordRef *ret_record);

TI_DLL_EXPORT int TI_API_CALL tie_Program_get_kernel_profiler_device_name(TieProgramRef self, TieStringHandle *ret_name);

TI_DLL_EXPORT int TI_API_CALL tie_Program_reinit_kernel_profiler_with_metrics(TieProgramRef self, const char **ap_metrics, size_t metrics_dim, bool *ret_b);

TI_DLL_EXPORT int TI_API_CALL tie_Program_kernel_profiler_total_time(TieProgramRef self, double *ret_time);

TI_DLL_EXPORT int TI_API_CALL tie_Program_set_kernel_profiler_toolkit(TieProgramRef self, const char *toolkit_name, bool *ret_b);

TI_DLL_EXPORT int TI_API_CALL tie_Program_timeline_clear(TieProgramRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Program_timeline_save(TieProgramRef self, const char *fn);

TI_DLL_EXPORT int TI_API_CALL tie_Program_print_memory_profiler_info(TieProgramRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Program_get_total_compilation_time(TieProgramRef self, double *ret_time);

TI_DLL_EXPORT int TI_API_CALL tie_Program_get_snode_num_dynamically_allocated(TieProgramRef self, TieSNodeRef snode, size_t *ret_size);

TI_DLL_EXPORT int TI_API_CALL tie_Program_materialize_runtime(TieProgramRef self);

TI_DLL_EXPORT int TI_API_CALL tie_Program_make_aot_module_builder(TieProgramRef self, int arch, const char **ap_caps, size_t caps_count, TieAotModuleBuilderHandle *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_Program_get_snode_tree_size(TieProgramRef self, int *ret_size);

TI_DLL_EXPORT int TI_API_CALL tie_Program_get_snode_root(TieProgramRef self, int tree_id, TieSNodeRef *ret_snode);

TI_DLL_EXPORT int TI_API_CALL tie_Program_create_kernel(TieProgramRef self, const char *name, int autodiff_mode, TieKernelRef *ret_kernel);

TI_DLL_EXPORT int TI_API_CALL tie_Program_create_function(TieProgramRef self, const char *func_name, int func_id, int instance_id, TieFunctionRef *ret_func);

TI_DLL_EXPORT int TI_API_CALL tie_Program_create_sparse_matrix(TieProgramRef self, int n, int m, TieDataTypeRef dtype, const char *storage_format, TieSparseMatrixHandle *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_Program_make_sparse_matrix_from_ndarray(TieProgramRef self, TieSparseMatrixRef sm, TieNdarrayRef ndarray);

TI_DLL_EXPORT int TI_API_CALL tie_Program_create_ndarray(TieProgramRef self, TieDataTypeRef dt, int *ap_shape, size_t shape_dim, int external_array_layout, bool zero_fill, TieNdarrayRef *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_Program_delete_ndarray(TieProgramRef self, TieNdarrayRef ndarray);

TI_DLL_EXPORT int TI_API_CALL tie_Program_create_texture(TieProgramRef self, int fmt, int *ap_shape, size_t shape_dim, TieTextureRef *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_Program_fill_ndarray_float(TieProgramRef self, TieNdarrayRef ndarray, float f);

TI_DLL_EXPORT int TI_API_CALL tie_Program_fill_ndarray_int(TieProgramRef self, TieNdarrayRef ndarray, int32_t i);

TI_DLL_EXPORT int TI_API_CALL tie_Program_fill_ndarray_uint(TieProgramRef self, TieNdarrayRef ndarray, uint32_t u);

TI_DLL_EXPORT int TI_API_CALL tie_Program_compile_kernel(TieProgramRef self, TieCompileConfigRef compile_config, TieKernelRef kernel, TieCompiledKernelDataRef *ret_ckd);

TI_DLL_EXPORT int TI_API_CALL tie_Program_launch_kernel(TieProgramRef self, TieCompiledKernelDataRef kernel_data, TieLaunchContextBuilderRef ctx);

// struct KernelProfileTracedRecord
TI_DLL_EXPORT int TI_API_CALL tie_KernelProfileTracedRecord_get_register_per_thread(TieKernelProfileTracedRecordRef self, int *ret_register_per_thread);

TI_DLL_EXPORT int TI_API_CALL tie_KernelProfileTracedRecord_get_shared_mem_per_block(TieKernelProfileTracedRecordRef self, int *ret_shared_mem_per_block);

TI_DLL_EXPORT int TI_API_CALL tie_KernelProfileTracedRecord_get_grid_size(TieKernelProfileTracedRecordRef self, int *ret_grid_size);

TI_DLL_EXPORT int TI_API_CALL tie_KernelProfileTracedRecord_get_block_size(TieKernelProfileTracedRecordRef self, int *ret_block_size);

TI_DLL_EXPORT int TI_API_CALL tie_KernelProfileTracedRecord_get_active_blocks_per_multiprocessor(TieKernelProfileTracedRecordRef self, int *ret_active_blocks_per_multiprocessor);

TI_DLL_EXPORT int TI_API_CALL tie_KernelProfileTracedRecord_get_kernel_elapsed_time_in_ms(TieKernelProfileTracedRecordRef self, float *ret_kernel_elapsed_time_in_ms);

TI_DLL_EXPORT int TI_API_CALL tie_KernelProfileTracedRecord_get_time_since_base(TieKernelProfileTracedRecordRef self, float *ret_time_since_base);

TI_DLL_EXPORT int TI_API_CALL tie_KernelProfileTracedRecord_get_name(TieKernelProfileTracedRecordRef self, const char **ret_name);

TI_DLL_EXPORT int TI_API_CALL tie_KernelProfileTracedRecord_get_num_metric_values(TieKernelProfileTracedRecordRef self, size_t *ret_size);

TI_DLL_EXPORT int TI_API_CALL tie_KernelProfileTracedRecord_get_metric_value(TieKernelProfileTracedRecordRef self, size_t index, float *ret_value);

// util functions (for Python)
TI_DLL_EXPORT int TI_API_CALL tie_G_set_pytype_tp_finalize(void *py_type_object);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
// clang-format on
