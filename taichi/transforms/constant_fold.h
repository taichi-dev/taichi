#pragma once

#include "taichi/ir/pass.h"

namespace taichi::lang {

class ConstantFoldPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    Program *program;
  };
};

}  // namespace taichi::lang
