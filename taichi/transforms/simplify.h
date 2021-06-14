#pragma once

#include "taichi/ir/pass.h"

namespace taichi {
namespace lang {

class FullSimplifyPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    bool after_lower_access;
    Program *program;
  };
};

}  // namespace lang
}  // namespace taichi
