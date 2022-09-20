#include "c_api_test_utils.h"
#include "taichi_llvm_impl.h"
#include "taichi/platform/cuda/detect_cuda.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif

#ifdef TI_WITH_OPENGL
#include "taichi/rhi/opengl/opengl_api.h"
#endif

namespace capi {
namespace utils {

bool is_vulkan_available() {
#ifdef TI_WITH_VULKAN
  return taichi::lang::vulkan::is_vulkan_api_available();
#else
  return false;
#endif
}

bool is_opengl_available() {
#ifdef TI_WITH_OPENGL
  return taichi::lang::opengl::is_opengl_api_available();
#else
  return false;
#endif
}

bool is_cuda_available() {
  return taichi::is_cuda_api_available();
}

void check_runtime_error(TiRuntime runtime) {
#ifdef TI_WITH_LLVM
  auto *llvm_runtime = dynamic_cast<capi::LlvmRuntime *>((Runtime *)runtime);
  if (!llvm_runtime) {
    TI_NOT_IMPLEMENTED;
  }
  llvm_runtime->check_runtime_error();
#else
  TI_NOT_IMPLEMENTED;
#endif
}

TiNdarrayAndMem make_ndarray(TiRuntime runtime,
                             TiDataType dtype,
                             const int *arr_shape,
                             int arr_dims,
                             const int *element_shape,
                             int element_dims,
                             bool host_read,
                             bool host_write) {
  size_t alloc_size = 1;
  if (dtype == TiDataType::TI_DATA_TYPE_F64 ||
      dtype == TiDataType::TI_DATA_TYPE_I64 ||
      dtype == TiDataType::TI_DATA_TYPE_U64) {
    alloc_size = 8;

  } else if (dtype == TiDataType::TI_DATA_TYPE_F32 ||
             dtype == TiDataType::TI_DATA_TYPE_I32 ||
             dtype == TiDataType::TI_DATA_TYPE_U32) {
    alloc_size = 4;

  } else if (dtype == TI_DATA_TYPE_F16 || dtype == TI_DATA_TYPE_I16 ||
             dtype == TI_DATA_TYPE_U16) {
    alloc_size = 2;

  } else if (dtype == TI_DATA_TYPE_I8 || dtype == TI_DATA_TYPE_U8) {
    alloc_size = 1;

  } else {
    TI_ASSERT(false);
  }

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

  TiNdArray arg_array{};
  arg_array.memory = res.memory_;
  arg_array.shape = shape;
  arg_array.elem_shape = e_shape;
  arg_array.elem_type = dtype;

  TiArgumentValue arg_value{};
  arg_value.ndarray = arg_array;

  res.arg_.type = TiArgumentType::TI_ARGUMENT_TYPE_NDARRAY;
  res.arg_.value = arg_value;

  return res;
}

}  // namespace utils
}  // namespace capi
