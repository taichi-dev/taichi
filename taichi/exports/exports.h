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
typedef TieHandle TieCompileConfigHandle;         // class CompileConfig
typedef TieHandle TieKernelHandle;                // class Kernel
typedef TieHandle TieFunctionHandle;              // class Function
typedef TieHandle TieNdarrayHandle;               // class NDArray
typedef TieHandle TieTextureHandle;               // class Texture
typedef TieHandle TieLaunchContextBuilderHandle;  // class LaunchContextBuilder
typedef TieHandle TieDataTypeHandle;              // class DataType
typedef TieHandle TieASTBuilderHandle;            // class ASTBuilder
typedef TieHandle TieSNodeHandle;                 // class SNode

// Error processing
TI_DLL_EXPORT int TI_API_CALL tie_G_set_last_error(int error, const char *msg);

TI_DLL_EXPORT int TI_API_CALL tie_G_get_last_error(int *ret_error, const char **ret_msg);

// Arch handling
TI_DLL_EXPORT int TI_API_CALL tie_G_arch_name(int arch, const char **ret_name);

TI_DLL_EXPORT int TI_API_CALL tie_G_arch_from_name(const char *name, int *ret_arch);

TI_DLL_EXPORT int TI_API_CALL tie_G_host_arch(int *ret_arch);

TI_DLL_EXPORT int TI_API_CALL tie_G_is_extension_supported(int arch, int extension, bool *ret_supported);

// default_compile_config handling
TI_DLL_EXPORT int TI_API_CALL tie_G_default_compile_config(TieCompileConfigHandle *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_G_reset_default_compile_config();

// class CompileConfig
TI_DLL_EXPORT int TI_API_CALL tie_CompileConfig_create(TieCompileConfigHandle *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_CompileConfig_destroy(TieCompileConfigHandle self);

#define TIE_DECL_COMPILE_CONFIG_GET_SET_ATTR(TaichiStruct, attr_name, attr_type, get_set_type) \
  TI_DLL_EXPORT int TI_API_CALL tie_CompileConfig_get_##attr_name(TieCompileConfigHandle self, get_set_type *ret_##attr_name); \
  TI_DLL_EXPORT int TI_API_CALL tie_CompileConfig_set_##attr_name(TieCompileConfigHandle self, get_set_type attr_name);
#define TIE_PER_COMPILE_CONFIG_ATTR TIE_DECL_COMPILE_CONFIG_GET_SET_ATTR
#include "inc/compile_config.inc.h"
#undef TIE_PER_COMPILE_CONFIG_ATTR
#undef TIE_DECL_COMPILE_CONFIG_GET_SET_ATTR

// class Kernel
TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_scalar_param(TieKernelHandle self, TieDataTypeHandle dt, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_arr_param(TieKernelHandle self, TieDataTypeHandle dt, int total_dim, int *ap_element_shape, size_t element_shape_dim, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_ndarray_param(TieKernelHandle self, TieDataTypeHandle dt, int ndim, const char *name, int needs_grad, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_texture_param(TieKernelHandle self, int total_dim, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_pointer_param(TieKernelHandle self, TieDataTypeHandle dt, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_rw_texture_param(TieKernelHandle self, int total_dim, int format, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_insert_ret(TieKernelHandle self, TieDataTypeHandle dt, int *ret_ret_index);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_finalize_rets(TieKernelHandle self);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_finalize_params(TieKernelHandle self);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_ast_builder(TieKernelHandle self, TieASTBuilderHandle *ret_ast_builder);

TI_DLL_EXPORT int TI_API_CALL tie_Kernel_no_activate(TieKernelHandle self, TieSNodeHandle snode);

// class Function
TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_scalar_param(TieFunctionHandle self, TieDataTypeHandle dt, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_arr_param(TieFunctionHandle self, TieDataTypeHandle dt, int total_dim, int *ap_element_shape, size_t element_shape_dim, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_ndarray_param(TieFunctionHandle self, TieDataTypeHandle dt, int ndim, const char *name, int needs_grad, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_texture_param(TieFunctionHandle self, int total_dim, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_pointer_param(TieFunctionHandle self, TieDataTypeHandle dt, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_rw_texture_param(TieFunctionHandle self, int total_dim, int format, const char *name, int *ret_param_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_set_function_body(TieFunctionHandle self, TieCallback func);

TI_DLL_EXPORT int TI_API_CALL tie_Function_insert_ret(TieFunctionHandle self, TieDataTypeHandle dt, int *ret_ret_index);

TI_DLL_EXPORT int TI_API_CALL tie_Function_finalize_rets(TieFunctionHandle self);

TI_DLL_EXPORT int TI_API_CALL tie_Function_finalize_params(TieFunctionHandle self);

TI_DLL_EXPORT int TI_API_CALL tie_Function_ast_builder(TieFunctionHandle self, TieASTBuilderHandle *ret_ast_builder);

// class LaunchContextBuilder
TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_create(TieKernelHandle kernel_handle, TieLaunchContextBuilderHandle *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_destroy(TieLaunchContextBuilderHandle self);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_int(TieLaunchContextBuilderHandle self, int arg_id, int64_t i64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_uint(TieLaunchContextBuilderHandle self, int arg_id, uint64_t u64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_float(TieLaunchContextBuilderHandle self, int arg_id, double d);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_struct_arg_int(TieLaunchContextBuilderHandle self, int *ap_arg_indices, size_t arg_indices_dim, int64_t i64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_struct_arg_uint(TieLaunchContextBuilderHandle self, int *ap_arg_indices, size_t arg_indices_dim, uint64_t u64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_struct_arg_float(TieLaunchContextBuilderHandle self, int *ap_arg_indices, size_t arg_indices_dim, double d);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_external_array_with_shape(TieLaunchContextBuilderHandle self, int arg_id, uintptr_t ptr, uint64_t size, int64_t *ap_shape, size_t shape_dim, uintptr_t grad_ptr);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_ndarray(TieLaunchContextBuilderHandle self, int arg_id, TieNdarrayHandle arr);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_ndarray_with_grad(TieLaunchContextBuilderHandle self, int arg_id, TieNdarrayHandle arr, TieNdarrayHandle arr_grad);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_texture(TieLaunchContextBuilderHandle self, int arg_id, TieTextureHandle tex);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_rw_texture(TieLaunchContextBuilderHandle self, int arg_id, TieTextureHandle tex);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_get_struct_ret_int(TieLaunchContextBuilderHandle self, int *ap_index, size_t index_dim, int64_t *ret_i64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_get_struct_ret_uint(TieLaunchContextBuilderHandle self, int *ap_index, size_t index_dim, uint64_t *ret_u64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_get_struct_ret_float(TieLaunchContextBuilderHandle self, int *ap_index, size_t index_dim, double *ret_d);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
// clang-format on
