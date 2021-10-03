#pragma once

#include "taichi/ir/pass.h"

namespace taichi {
namespace lang {

class InliningPass : public Pass {
 public:
  static const PassID id;

  struct Args {};
};

}  // namespace lang
}  // namespace taichi
