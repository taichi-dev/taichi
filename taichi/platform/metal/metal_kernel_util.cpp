#include "metal_kernel_util.h"

#define TI_RUNTIME_HOST
#include <taichi/runtime/context.h>
#undef TI_RUNTIME_HOST

TLANG_NAMESPACE_BEGIN

namespace metal {

MetalKernelArgsAttributes::MetalKernelArgsAttributes(
    const std::vector<Kernel::Arg>& args)
    : args_bytes_(0), extra_args_bytes_(Context::extra_args_size) {
  arg_attribs_vec_.reserve(args.size());
  for (const auto& ka : args) {
    ArgAttributes ma;
    ma.dt = to_metal_type(ka.dt);
    const size_t dt_bytes = metal_data_type_bytes(ma.dt);
    if (dt_bytes > 4) {
      // Metal doesn't support 64bit data buffers.
      TI_ERROR("Metal kernel only supports <= 32-bit data, got {}",
               metal_data_type_name(ma.dt));
    }
    ma.is_array = ka.is_nparray;
    ma.stride = ma.is_array ? ka.size : dt_bytes;
    ma.index = arg_attribs_vec_.size();
    ma.is_return_val = ka.is_return_value;
    arg_attribs_vec_.push_back(ma);
  }

  std::vector<int> scalar_indices;
  std::vector<int> array_indices;
  for (int i = 0; i < arg_attribs_vec_.size(); ++i) {
    if (arg_attribs_vec_[i].is_array) {
      array_indices.push_back(i);
    } else {
      scalar_indices.push_back(i);
    }
  }
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
}

}  // namespace metal

TLANG_NAMESPACE_END
