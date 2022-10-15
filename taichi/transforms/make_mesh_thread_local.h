#pragma once

#include "taichi/ir/pass.h"

namespace taichi::lang {

class MakeMeshThreadLocal : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
  };
};

}  // namespace taichi::lang
