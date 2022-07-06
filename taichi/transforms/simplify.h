#pragma once

#include "taichi/ir/pass.h"

namespace taichi {
namespace lang {

class FullSimplifyPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    bool after_lower_access;
    // Switch off some optimization in store forwarding if there is an autodiff
    // pass after the full_simplify
    bool with_autodiff_after;
    Program *program;
  };
};

}  // namespace lang
}  // namespace taichi
