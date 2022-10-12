#pragma once

#include "taichi/ir/pass.h"

namespace taichi::lang {

class DemoteMeshStatements : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
  };
};

}  // namespace taichi::lang
