#include "taichi/program/function.h"
#include "taichi/program/program.h"
#include "taichi/ir/transforms.h"
#include "taichi/analysis/offline_cache_util.h"

namespace taichi::lang {

Function::Function(Program *program, const FunctionKey &func_key)
    : func_key(func_key) {
  this->program = program;
}

void Function::set_function_body(const std::function<void()> &func) {
  context = std::make_unique<FrontendContext>();
  ir = context->get_root();
  ir_start_from_ast_ = true;

  func();
}

void Function::set_function_body(std::unique_ptr<IRNode> func_body) {
  ir = std::move(func_body);
  ir_start_from_ast_ = false;
}

std::string Function::get_name() const {
  return func_key.get_full_name();
}

}  // namespace taichi::lang
