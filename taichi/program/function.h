#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/ir.h"

namespace taichi {
namespace lang {

class Program;

class Function {
 public:
  Program *prog;
  std::string funcid;
  std::unique_ptr<IRNode> ir;

  Function(Program *prog, const std::string &funcid);

  void set_function_body(const std::function<void()> &func);
};

}  // namespace lang
}  // namespace taichi
