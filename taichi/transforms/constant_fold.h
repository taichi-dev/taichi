#pragma once

#include "taichi/ir/pass.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

class ConstantFoldPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    Program *program;
  };
};

}  // namespace lang
}  // namespace taichi
