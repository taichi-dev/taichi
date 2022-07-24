#pragma once

#include <vector>

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"

namespace taichi {
namespace lang {

// TODO: It would be better if this can be unified with Callable::Arg. However,
// Callable::Arg is not easily serializable.
struct LlvmLaunchArgInfo {
  bool is_array{false};

  TI_IO_DEF(is_array);

  bool operator==(const LlvmLaunchArgInfo &other) const;
  bool operator!=(const LlvmLaunchArgInfo &other) const {
    return !(*this == other);
  }
};

class Kernel;

std::vector<LlvmLaunchArgInfo> infer_launch_args(const Kernel *kernel);

}  // namespace lang
}  // namespace taichi
