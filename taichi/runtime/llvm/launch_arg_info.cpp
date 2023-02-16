#include "taichi/runtime/llvm/launch_arg_info.h"

#include "taichi/program/kernel.h"

namespace taichi::lang {

bool LlvmLaunchArgInfo::operator==(const LlvmLaunchArgInfo &other) const {
  return is_array == other.is_array;
}

std::vector<LlvmLaunchArgInfo> infer_launch_args(const Kernel *kernel) {
  std::vector<LlvmLaunchArgInfo> res;
  res.reserve(kernel->parameter_list.size());
  for (const auto &a : kernel->parameter_list) {
    res.push_back(LlvmLaunchArgInfo{a.is_ptr});
  }
  return res;
}

}  // namespace taichi::lang
