#pragma once

#include "taichi/ir/pass.h"

namespace taichi::lang {

class InliningPass : public Pass {
 public:
  static const PassID id;

  struct Args {};
};

}  // namespace taichi::lang
