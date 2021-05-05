#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/ir.h"
#include "taichi/program/kernel.h"

namespace taichi {
namespace lang {

class Program;

// TODO: Let Function and Kernel inherit from some class like "Callable"
//  and merge the duplicated part?
class Function {
 public:
  Program *prog;
  std::string funcid;
  std::unique_ptr<IRNode> ir;
  using Arg = Kernel::Arg;
  using Ret = Kernel::Ret;

  std::vector<Arg> args;
  std::vector<Ret> rets;

  Function(Program *prog, const std::string &funcid);

  void set_function_body(const std::function<void()> &func);

  int insert_arg(DataType dt, bool is_external_array);

  int insert_ret(DataType dt);
};

}  // namespace lang
}  // namespace taichi
