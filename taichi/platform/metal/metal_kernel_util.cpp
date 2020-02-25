#include "metal_kernel_util.h"

#define TI_RUNTIME_HOST
#include <taichi/context.h>
#undef TI_RUNTIME_HOST

TLANG_NAMESPACE_BEGIN

namespace metal {

int MetalKernelArgsAttributes::insert_arg(DataType dt,
                                         bool is_array,
                                         size_t size,
                                         bool is_return_val) {
  ArgAttributes a;
  a.dt = to_metal_type(dt);
  const size_t dt_bytes = metal_data_type_bytes(a.dt);
  if (dt_bytes > 4) {
    // Metal doesn't support 64bit data buffers.
    // TODO(k-ye): See if Metal supports less-than-32bit data buffers.
    TI_ERROR("Metal kernel only supports <= 32-bit data, got {}",
            metal_data_type_name(a.dt));
  }
  a.is_array = is_array;
  a.stride = is_array ? size : dt_bytes;
  a.index = arg_attribs_vec_.size();
  a.is_return_val = is_return_val;
  arg_attribs_vec_.push_back(a);
  return a.index;
}

void MetalKernelArgsAttributes::finalize() {
  std::vector<int> scalar_indices;
  std::vector<int> array_indices;
  for (int i = 0; i < arg_attribs_vec_.size(); ++i) {
    if (arg_attribs_vec_[i].is_array) {
      array_indices.push_back(i);
    } else {
      scalar_indices.push_back(i);
    }
  }
  args_bytes_ = 0;
  // Put scalar args in the memory first
  for (int i : scalar_indices) {
    auto& arg = arg_attribs_vec_[i];
    arg.offset_in_mem = args_bytes_;
    args_bytes_ += arg.stride;
  }
  // Then the array args
  for (int i : array_indices) {
    auto& arg = arg_attribs_vec_[i];
    arg.offset_in_mem = args_bytes_;
    args_bytes_ += arg.stride;
  }
  extra_args_bytes_ = Context::extra_args_size;
}

}  // namespace metal

TLANG_NAMESPACE_END
