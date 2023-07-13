#include "exports.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <exception>

#include "taichi/common/exceptions.h"
#include "taichi/program/kernel.h"
#include "taichi/program/launch_context_builder.h"

#define TIE_WRAP_ERROR_MSG(msg)                         \
  std::string("Tie-API error (")                        \
      .append(__func__)                                 \
      .append("): " msg " (triggered at " __FILE__ ":") \
      .append(std::to_string(__LINE__))                 \
      .append(")")

#define TIE_CHECK_PTR_NOT_NULL(ptr)                            \
  if (ptr == nullptr) {                                        \
    tie_api_set_last_error_impl(                               \
        TIE_ERROR_INVALID_ARGUMENT,                            \
        TIE_WRAP_ERROR_MSG("Argument '" #ptr "' is nullptr")); \
    return TIE_ERROR_INVALID_ARGUMENT;                         \
  }

#define TIE_CHECK_RETURN_ARG(ret_param)                                     \
  if (ret_param == nullptr) {                                               \
    tie_api_set_last_error_impl(                                            \
        TIE_ERROR_INVALID_RETURN_ARG,                                       \
        TIE_WRAP_ERROR_MSG("Return argument '" #ret_param "' is nullptr")); \
    return TIE_ERROR_INVALID_RETURN_ARG;                                    \
  }

#define TIE_CHECK_HANDLE(handle)                                \
  if (handle == nullptr) {                                      \
    tie_api_set_last_error_impl(                                \
        TIE_ERROR_INVALID_HANDLE,                               \
        TIE_WRAP_ERROR_MSG("Handle '" #handle "' is nullptr")); \
    return TIE_ERROR_INVALID_HANDLE;                            \
  }

#define TIE_FUNCTION_BODY_BEGIN() try {
#define TIE_FUNCTION_BODY_END()                                              \
  }                                                                          \
  catch (const taichi::lang::TaichiTypeError &e) {                           \
    tie_api_set_last_error_impl(TIE_ERROR_TAICHI_TYPE_ERROR, e.what());      \
    return TIE_ERROR_TAICHI_TYPE_ERROR;                                      \
  }                                                                          \
  catch (const taichi::lang::TaichiSyntaxError &e) {                         \
    tie_api_set_last_error_impl(TIE_ERROR_TAICHI_SYNTAX_ERROR, e.what());    \
    return TIE_ERROR_TAICHI_SYNTAX_ERROR;                                    \
  }                                                                          \
  catch (const taichi::lang::TaichiIndexError &e) {                          \
    tie_api_set_last_error_impl(TIE_ERROR_TAICHI_INDEX_ERROR, e.what());     \
    return TIE_ERROR_TAICHI_INDEX_ERROR;                                     \
  }                                                                          \
  catch (const taichi::lang::TaichiRuntimeError &e) {                        \
    tie_api_set_last_error_impl(TIE_ERROR_TAICHI_RUNTIME_ERROR, e.what());   \
    return TIE_ERROR_TAICHI_RUNTIME_ERROR;                                   \
  }                                                                          \
  catch (const taichi::lang::TaichiAssertionError &e) {                      \
    tie_api_set_last_error_impl(TIE_ERROR_TAICHI_ASSERTION_ERROR, e.what()); \
    return TIE_ERROR_TAICHI_ASSERTION_ERROR;                                 \
  }                                                                          \
  catch (const std::bad_alloc &e) {                                          \
    tie_api_set_last_error_impl(TIE_ERROR_OUT_OF_MEMORY, e.what());          \
    return TIE_ERROR_OUT_OF_MEMORY;                                          \
  }                                                                          \
  catch (const std::exception &e) {                                          \
    tie_api_set_last_error_impl(TIE_ERROR_UNKNOWN_CXX_EXCEPTION, e.what());  \
    return TIE_ERROR_UNKNOWN_CXX_EXCEPTION;                                  \
  }                                                                          \
  catch (...) {                                                              \
    tie_api_set_last_error_impl(TIE_ERROR_UNKNOWN_CXX_EXCEPTION,             \
                                "C++ Exception");                            \
    return TIE_ERROR_UNKNOWN_CXX_EXCEPTION;                                  \
  }                                                                          \
  return TIE_ERROR_SUCCESS

namespace {

struct TieErrorCache {
  int error{TIE_ERROR_SUCCESS};
  std::string msg;
};

static thread_local TieErrorCache tie_last_error;

void tie_api_set_last_error_impl(int error, std::string msg) {
  tie_last_error.error = error;
  tie_last_error.msg = std::move(msg);
}

}  // namespace

int tie_G_set_last_error(int error, const char *msg) {
  tie_api_set_last_error_impl(error, msg ? msg : "");
  return TIE_ERROR_SUCCESS;
}

int tie_G_get_last_error(int *ret_error, const char **ret_msg) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_RETURN_ARG(ret_error);
  TIE_CHECK_RETURN_ARG(ret_msg);
  *ret_error = tie_last_error.error;
  *ret_msg = tie_last_error.msg.c_str();
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_create(TieKernelHandle kernel_handle,
                                    TieLaunchContextBuilderHandle *ret_handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_RETURN_ARG(ret_handle);
  TIE_CHECK_HANDLE(kernel_handle);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(kernel_handle);
  auto *builder = new taichi::lang::LaunchContextBuilder(kernel);
  *ret_handle = reinterpret_cast<TieLaunchContextBuilderHandle>(builder);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_destroy(TieLaunchContextBuilderHandle handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  delete builder;
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_int(TieLaunchContextBuilderHandle handle,
                                         int arg_id,
                                         int64_t i64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  builder->set_arg_int(arg_id, i64);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_uint(TieLaunchContextBuilderHandle handle,
                                          int arg_id,
                                          uint64_t u64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  builder->set_arg_uint(arg_id, u64);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_float(TieLaunchContextBuilderHandle handle,
                                           int arg_id,
                                           double d) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  builder->set_arg_float(arg_id, d);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_struct_arg_int(
    TieLaunchContextBuilderHandle handle,
    int *arg_indices,
    size_t arg_indices_dim,
    int64_t i64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(arg_indices);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(arg_indices, arg_indices + arg_indices_dim);
  builder->set_struct_arg(indices, i64);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_struct_arg_uint(
    TieLaunchContextBuilderHandle handle,
    int *arg_indices,
    size_t arg_indices_dim,
    uint64_t u64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(arg_indices);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(arg_indices, arg_indices + arg_indices_dim);
  builder->set_struct_arg(indices, u64);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_struct_arg_float(
    TieLaunchContextBuilderHandle handle,
    int *arg_indices,
    size_t arg_indices_dim,
    double d) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(arg_indices);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(arg_indices, arg_indices + arg_indices_dim);
  builder->set_struct_arg(indices, d);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_external_array_with_shape(
    TieLaunchContextBuilderHandle handle,
    int arg_id,
    uintptr_t ptr,
    uint64_t size,
    int64_t *shape,
    size_t shape_dim,
    uintptr_t grad_ptr) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(shape);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int64_t> shape_vec(shape, shape + shape_dim);
  builder->set_arg_external_array_with_shape(arg_id, ptr, size, shape_vec,
                                             grad_ptr);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_ndarray(
    TieLaunchContextBuilderHandle handle,
    int arg_id,
    TieNdarrayHandle arr) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(arr);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *ndarray = reinterpret_cast<const taichi::lang::Ndarray *>(arr);
  builder->set_arg_ndarray(arg_id, *ndarray);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_ndarray_with_grad(
    TieLaunchContextBuilderHandle handle,
    int arg_id,
    TieNdarrayHandle arr,
    TieNdarrayHandle arr_grad) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(arr);
  TIE_CHECK_HANDLE(arr_grad);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *ndarray = reinterpret_cast<const taichi::lang::Ndarray *>(arr);
  auto *ndarray_grad =
      reinterpret_cast<const taichi::lang::Ndarray *>(arr_grad);
  builder->set_arg_ndarray_with_grad(arg_id, *ndarray, *ndarray_grad);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_texture(
    TieLaunchContextBuilderHandle handle,
    int arg_id,
    TieTextureHandle tex) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(tex);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *texture = reinterpret_cast<const taichi::lang::Texture *>(tex);
  builder->set_arg_texture(arg_id, *texture);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_rw_texture(
    TieLaunchContextBuilderHandle handle,
    int arg_id,
    TieTextureHandle tex) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(tex);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *texture = reinterpret_cast<const taichi::lang::Texture *>(tex);
  builder->set_arg_rw_texture(arg_id, *texture);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_get_struct_ret_int(
    TieLaunchContextBuilderHandle handle,
    int *index,
    size_t index_dim,
    int64_t *ret_i64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(index);
  TIE_CHECK_RETURN_ARG(ret_i64);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(index, index + index_dim);
  *ret_i64 = builder->get_struct_ret_int(indices);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_get_struct_ret_uint(
    TieLaunchContextBuilderHandle handle,
    int *index,
    size_t index_dim,
    uint64_t *ret_u64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(index);
  TIE_CHECK_RETURN_ARG(ret_u64);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(index, index + index_dim);
  *ret_u64 = builder->get_struct_ret_uint(indices);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_get_struct_ret_float(
    TieLaunchContextBuilderHandle handle,
    int *index,
    size_t index_dim,
    double *ret_d) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(index);
  TIE_CHECK_RETURN_ARG(ret_d);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(index, index + index_dim);
  *ret_d = builder->get_struct_ret_float(indices);
  TIE_FUNCTION_BODY_END();
}
