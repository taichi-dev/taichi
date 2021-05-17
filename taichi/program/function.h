#pragma once

#include "taichi/program/callable.h"
#include "taichi/program/function_key.h"

namespace taichi {
namespace lang {

class Function : public Callable {
 public:
  FunctionKey func_key;

  std::vector<Arg> args;
  std::vector<Ret> rets;

  Function(Program *program, const FunctionKey &func_key);

  // Set the function body to a frontend Python function which generates the C++
  // AST.
  void set_function_body(const std::function<void()> &func);

  // Set the function body to a CHI IR.
  void set_function_body(std::unique_ptr<IRNode> func_body);
};

}  // namespace lang
}  // namespace taichi
