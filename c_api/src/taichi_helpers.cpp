#include <assert.h>
#include "taichi/taichi_helpers.h"

TiNdarrayAndMem make_ndarray(TiRuntime runtime,
                             TiDataType dtype,
                             const int *arr_shape,
                             int arr_dims,
                             const int *element_shape,
                             int element_dims,
                             bool host_read,
                             bool host_write) {
  size_t alloc_size = 4;
  assert(dtype == TiDataType::TI_DATA_TYPE_F32 ||
         dtype == TiDataType::TI_DATA_TYPE_I32 ||
         dtype == TiDataType::TI_DATA_TYPE_U32);

  for (int i = 0; i < arr_dims; i++) {
    alloc_size *= arr_shape[i];
  }

  for (int i = 0; i < element_dims; i++) {
    alloc_size *= element_shape[i];
  }

  TiNdarrayAndMem res;
  res.runtime_ = runtime;

  TiMemoryAllocateInfo alloc_info;
  alloc_info.size = alloc_size;
  alloc_info.host_write = host_write;
  alloc_info.host_read = host_read;
  alloc_info.export_sharing = false;
  alloc_info.usage = TiMemoryUsageFlagBits::TI_MEMORY_USAGE_STORAGE_BIT;

  res.memory_ = ti_allocate_memory(res.runtime_, &alloc_info);

  TiNdShape shape;
  shape.dim_count = (uint32_t)arr_dims;
  for (size_t i = 0; i < shape.dim_count; i++) {
    shape.dims[i] = arr_shape[i];
  }

  TiNdShape e_shape;
  e_shape.dim_count = (uint32_t)element_dims;
  for (size_t i = 0; i < e_shape.dim_count; i++) {
    e_shape.dims[i] = element_shape[i];
  }

  TiNdArray arg_array = {.memory = res.memory_,
                         .shape = shape,
                         .elem_shape = e_shape,
                         .elem_type = dtype};

  TiArgumentValue arg_value = {.ndarray = arg_array};

  res.arg_ = {.type = TiArgumentType::TI_ARGUMENT_TYPE_NDARRAY,
              .value = arg_value};

  return res;
}
