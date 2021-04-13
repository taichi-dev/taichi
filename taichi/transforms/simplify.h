#pragma once

#include "taichi/ir/pass.h"

namespace taichi {
namespace lang {

class SimplifyPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    Kernel *kernel;
  };
};

class FullSimplifyPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    bool after_lower_access;
    Kernel *kernel;
  };
};

}  // namespace lang
}  // namespace taichi
