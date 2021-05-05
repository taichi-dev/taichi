#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/ir.h"
#include "taichi/program/arch.h"

namespace taichi {
namespace lang {

class Function {
 public:
  std::string funcid;
  std::unique_ptr<IRNode> ir;

  explicit Function(const std::string &funcid);

  void set_function_body(const std::function<void()> &func);
};

}  // namespace lang
}  // namespace taichi
