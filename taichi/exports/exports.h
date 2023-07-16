// clang-format off
#pragma once

#if defined(TI_EXPORTS_TO_PY)
#define _Atomic(x) x  // To make pycparser happy
#endif // defined(TI_EXPORTS_TO_PY)

#include <stddef.h>
#include <stdint.h>

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

  // Unknown error
  TIE_ERROR_UNKNOWN = -0x7FFFFFFF // INT_MIN
} TieError;

typedef void *TieHandle;

// Error processing
TI_DLL_EXPORT int TI_API_CALL tie_G_set_last_error(int error, const char *msg);

TI_DLL_EXPORT int TI_API_CALL tie_G_get_last_error(int *ret_error, const char **ret_msg);

// class Kernel
typedef TieHandle TieKernelHandle;

// class Ndarray
typedef TieHandle TieNdarrayHandle;

// class Texture
typedef TieHandle TieTextureHandle;

// class LaunchContextBuilder
typedef TieHandle TieLaunchContextBuilderHandle;

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
