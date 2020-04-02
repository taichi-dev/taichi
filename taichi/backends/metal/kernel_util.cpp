#include "taichi/backends/metal/kernel_util.h"

#include <unordered_map>

#define TI_RUNTIME_HOST
#include "taichi/runtime/llvm/context.h"
#undef TI_RUNTIME_HOST

TLANG_NAMESPACE_BEGIN

namespace metal {

// static
std::string KernelAttributes::buffers_name(Buffers b) {
#define REGISTER_NAME(x) \
  { Buffers::x, #x }
  const static std::unordered_map<Buffers, std::string> m = {
      REGISTER_NAME(Root),
      REGISTER_NAME(GlobalTmps),
      REGISTER_NAME(Args),
      REGISTER_NAME(Runtime),
  };
#undef REGISTER_NAME
  return m.find(b)->second;
}

std::string KernelAttributes::debug_string() const {
  std::string result;
  result += fmt::format(
      "<KernelAttributes name={} num_threads={} task_type={} buffers=[ ", name,
      num_threads, OffloadedStmt::task_type_name(task_type));
  for (auto b : buffers) {
    result += buffers_name(b) + " ";
  }
  result += "]";  // closes |buffers|
  // TODO(k-ye): show range_for
  if (task_type == OffloadedStmt::TaskType::clear_list ||
      task_type == OffloadedStmt::TaskType::listgen) {
    result += fmt::format(" snode={}", runtime_list_op_attribs.snode->id);
  }
  result += ">";
  return result;
}

KernelArgsAttributes::KernelArgsAttributes(const std::vector<Kernel::Arg> &args)
    : args_bytes_(0), extra_args_bytes_(Context::extra_args_size) {
  arg_attribs_vec_.reserve(args.size());
  for (const auto &ka : args) {
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
    auto &arg = arg_attribs_vec_[i];
    arg.offset_in_mem = args_bytes_;
    args_bytes_ += arg.stride;
  }
  // Then the array args
  for (int i : array_indices) {
    auto &arg = arg_attribs_vec_[i];
    arg.offset_in_mem = args_bytes_;
    args_bytes_ += arg.stride;
  }
}

}  // namespace metal

TLANG_NAMESPACE_END
