#pragma once

#include "taichi/ir/pass.h"

namespace taichi::lang {

class MakeBlockLocalPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
    bool verbose;
  };
};

}  // namespace taichi::lang
