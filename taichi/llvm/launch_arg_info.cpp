#include "taichi/llvm/launch_arg_info.h"

#include "taichi/program/kernel.h"

namespace taichi {
namespace lang {

std::vector<LlvmLaunchArgInfo> infer_launch_args(const Kernel *kernel) {
  std::vector<LlvmLaunchArgInfo> res;
  res.reserve(kernel->args.size());
  for (const auto &a : kernel->args) {
    res.push_back(LlvmLaunchArgInfo{a.is_array});
  }
  return res;
}

}  // namespace lang
}  // namespace taichi
