// clang-format off
#pragma once

#include <stdint.h>

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

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
//         2. Return type is always int, which is the error code (TieError).
//         3. Return value (except error code) is always the last parameters. And the parameter name always starts with "ret_".
//     1. Create Object: int TI_API_CALL tie_<ClassName>_create(<Params...>, Tie<ClassName>Handle *ret_handle);
//     2. Destroy Object: int TI_API_CALL tie_<ClassName>_destroy(Tie<ClassName>Handle handle);
//     3. Member Function: int TI_API_CALL tie_<ClassName>_<MemFuncName>(Tie<ClassName>Handle handle, <Params...>, <RetType> *ret_<RetName>);
//     4. Static Function: int TI_API_CALL tie_<ClassName>_<StaticFuncName>(<Params...>, <RetType> *ret_<RetName>);
//     5. Global Function: int TI_API_CALL tie_<GlobalFuncName>(<Params...>, <RetType> *ret_<RetName>);

typedef enum TieError {
  TIE_ERROR_SUCCESS = 0,
  TIE_ERROR_INVALID_RETURN_PARAM = -1,
  TIE_ERROR_INVALID_HANDLE = -2,
  TIE_ERROR_UNKNOWN = 0xFFFFFFF
} TieError;

TI_DLL_EXPORT int TI_API_CALL ticore_hello_world(const char *extra_msg);

typedef void * TieHandle;

// class Kernel
typedef TieHandle TieKernelHandle;

// class Ndarray
typedef TieHandle TieNdarrayHandle;

// class Texture
typedef TieHandle TieTextureHandle;

// class LaunchContextBuilder
typedef TieHandle TieLaunchContextBuilderHandle;

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_create(TieKernelHandle kernel_handle, TieLaunchContextBuilderHandle *ret_handle);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_destroy(TieLaunchContextBuilderHandle handle);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_int(TieLaunchContextBuilderHandle handle, int arg_id, int64_t i64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_uint(TieLaunchContextBuilderHandle handle, int arg_id, uint64_t u64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_float(TieLaunchContextBuilderHandle handle, int arg_id, double d);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_struct_arg_int(TieLaunchContextBuilderHandle handle, int *arg_indices, size_t arg_indices_dim, int64_t i64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_struct_arg_uint(TieLaunchContextBuilderHandle handle, int *arg_indices, size_t arg_indices_dim, uint64_t u64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_struct_arg_float(TieLaunchContextBuilderHandle handle, int *arg_indices, size_t arg_indices_dim, double d);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_external_array_with_shape(TieLaunchContextBuilderHandle handle, int arg_id, uintptr_t ptr, uint64_t size, int64_t *shape, size_t shape_dim, uintptr_t grad_ptr);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_ndarray(TieLaunchContextBuilderHandle handle, int arg_id, TieNdarrayHandle arr);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_ndarray_with_grad(TieLaunchContextBuilderHandle handle, int arg_id, TieNdarrayHandle arr, TieNdarrayHandle arr_grad);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_texture(TieLaunchContextBuilderHandle handle, int arg_id, TieTextureHandle tex);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_set_arg_rw_texture(TieLaunchContextBuilderHandle handle, int arg_id, TieTextureHandle tex);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_get_struct_ret_int(TieLaunchContextBuilderHandle handle, int *index, size_t index_dim, int64_t *ret_i64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_get_struct_ret_uint(TieLaunchContextBuilderHandle handle, int *index, size_t index_dim, uint64_t *ret_u64);

TI_DLL_EXPORT int TI_API_CALL tie_LaunchContextBuilder_get_struct_ret_float(TieLaunchContextBuilderHandle handle, int *index, size_t index_dim, double *ret_d);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
// clang-format on
