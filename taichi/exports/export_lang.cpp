#include "exports.h"

#include <cstdio>
#include <cstring>

#include "taichi/program/kernel.h"
#include "taichi/program/launch_context_builder.h"

#define TIE_CHECK_RET_PARAM(ret_param)     \
  if (ret_param == NULL) {                 \
    return TIE_ERROR_INVALID_RETURN_PARAM; \
  }

#define TIE_CHECK_HANDLE(handle)     \
  if (handle == NULL) {              \
    return TIE_ERROR_INVALID_HANDLE; \
  }


int tie_LaunchContextBuilder_create(TieKernelHandle kernel_handle,
                                    TieLaunchContextBuilderHandle *ret_handle) {
  TIE_CHECK_RET_PARAM(ret_handle);
  TIE_CHECK_HANDLE(kernel_handle);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(kernel_handle);
  auto *builder = new taichi::lang::LaunchContextBuilder(kernel);
  *ret_handle = reinterpret_cast<TieLaunchContextBuilderHandle>(builder);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_destroy(TieLaunchContextBuilderHandle handle) {
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  delete builder;
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_set_arg_int(TieLaunchContextBuilderHandle handle,
                                         int arg_id,
                                         int64_t i64) {
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  builder->set_arg_int(arg_id, i64);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_set_arg_uint(TieLaunchContextBuilderHandle handle,
                                          int arg_id,
                                          uint64_t u64) {
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  builder->set_arg_uint(arg_id, u64);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_set_arg_float(TieLaunchContextBuilderHandle handle,
                                           int arg_id,
                                           double d) {
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  builder->set_arg_float(arg_id, d);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_set_struct_arg_int(
    TieLaunchContextBuilderHandle handle,
    int *arg_indices,
    size_t arg_indices_dim,
    int64_t i64) {
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(arg_indices, arg_indices + arg_indices_dim);
  builder->set_struct_arg(indices, i64);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_set_struct_arg_uint(
    TieLaunchContextBuilderHandle handle,
    int *arg_indices,
    size_t arg_indices_dim,
    uint64_t u64) {
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(arg_indices, arg_indices + arg_indices_dim);
  builder->set_struct_arg(indices, u64);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_set_struct_arg_float(
    TieLaunchContextBuilderHandle handle,
    int *arg_indices,
    size_t arg_indices_dim,
    double d) {
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(arg_indices, arg_indices + arg_indices_dim);
  builder->set_struct_arg(indices, d);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_set_arg_external_array_with_shape(
    TieLaunchContextBuilderHandle handle,
    int arg_id,
    uintptr_t ptr,
    uint64_t size,
    int64_t *shape,
    size_t shape_dim,
    uintptr_t grad_ptr) {
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int64_t> shape_vec(shape, shape + shape_dim);
  builder->set_arg_external_array_with_shape(arg_id, ptr, size, shape_vec,
                                             grad_ptr);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_set_arg_ndarray(
    TieLaunchContextBuilderHandle handle,
    int arg_id,
    TieNdarrayHandle arr) {
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(arr);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *ndarray = reinterpret_cast<const taichi::lang::Ndarray *>(arr);
  builder->set_arg_ndarray(arg_id, *ndarray);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_set_arg_ndarray_with_grad(
    TieLaunchContextBuilderHandle handle,
    int arg_id,
    TieNdarrayHandle arr,
    TieNdarrayHandle arr_grad) {
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(arr);
  TIE_CHECK_HANDLE(arr_grad);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *ndarray = reinterpret_cast<const taichi::lang::Ndarray *>(arr);
  auto *ndarray_grad =
      reinterpret_cast<const taichi::lang::Ndarray *>(arr_grad);
  builder->set_arg_ndarray_with_grad(arg_id, *ndarray, *ndarray_grad);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_set_arg_texture(
    TieLaunchContextBuilderHandle handle,
    int arg_id,
    TieTextureHandle tex) {
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(tex);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *texture = reinterpret_cast<const taichi::lang::Texture *>(tex);
  builder->set_arg_texture(arg_id, *texture);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_set_arg_rw_texture(
    TieLaunchContextBuilderHandle handle,
    int arg_id,
    TieTextureHandle tex) {
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(tex);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *texture = reinterpret_cast<const taichi::lang::Texture *>(tex);
  builder->set_arg_rw_texture(arg_id, *texture);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_get_struct_ret_int(
    TieLaunchContextBuilderHandle handle,
    int *index,
    size_t index_dim,
    int64_t *ret_i64) {
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_RET_PARAM(ret_i64);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(index, index + index_dim);
  *ret_i64 = builder->get_struct_ret_int(indices);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_get_struct_ret_uint(
    TieLaunchContextBuilderHandle handle,
    int *index,
    size_t index_dim,
    uint64_t *ret_u64) {
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_RET_PARAM(ret_u64);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(index, index + index_dim);
  *ret_u64 = builder->get_struct_ret_uint(indices);
  return TIE_ERROR_SUCCESS;
}

int tie_LaunchContextBuilder_get_struct_ret_float(
    TieLaunchContextBuilderHandle handle,
    int *index,
    size_t index_dim,
    double *ret_d) {
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_RET_PARAM(ret_d);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(index, index + index_dim);
  *ret_d = builder->get_struct_ret_float(indices);
  return TIE_ERROR_SUCCESS;
}
