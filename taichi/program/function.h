#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/ir.h"
#include "taichi/program/function_key.h"
#include "taichi/program/kernel.h"

namespace taichi {
namespace lang {

class Program;

// TODO: Let Function and Kernel inherit from some class like "Callable"
//  and merge the common part?
class Function {
 public:
  Program *program;
  FunctionKey func_key;
  std::unique_ptr<IRNode> ir;
  using Arg = Kernel::Arg;
  using Ret = Kernel::Ret;

  std::vector<Arg> args;
  std::vector<Ret> rets;

  Function(Program *program, const FunctionKey &func_key);

  // Set the function body to a frontend Python function which generates the C++
  // AST.
  void set_function_body(const std::function<void()> &func);

  // Set the function body to a CHI IR.
  void set_function_body(std::unique_ptr<IRNode> func_body);

  int insert_arg(DataType dt, bool is_external_array);

  int insert_ret(DataType dt);
};

}  // namespace lang
}  // namespace taichi
