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
  context = std::make_unique<FrontendContext>(program->compile_config().arch);
  ir = context->get_root();
  ir_type_ = IRType::AST;

  func();

  if (program->compile_config().offline_cache) {  // For generating AST-Key
    std::ostringstream oss;
    gen_offline_cache_key(program, ir.get(), &oss);
    ast_serialization_data_ = oss.str();
  }
}

void Function::set_function_body(std::unique_ptr<IRNode> func_body) {
  ir = std::move(func_body);
  ir_type_ = IRType::InitialIR;
}

std::string Function::get_name() const {
  return func_key.get_full_name();
}

}  // namespace taichi::lang
