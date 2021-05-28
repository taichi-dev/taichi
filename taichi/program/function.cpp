#include "taichi/program/function.h"
#include "taichi/program/program.h"
#include "taichi/ir/transforms.h"

namespace taichi {
namespace lang {

Function::Function(Program *program, const FunctionKey &func_key)
    : func_key(func_key) {
  this->program = program;
}

void Function::set_function_body(const std::function<void()> &func) {
  // Do not corrupt the context calling this function here
  auto backup_context = std::move(taichi::lang::context);

  taichi::lang::context = std::make_unique<FrontendContext>();
  ir = taichi::lang::context->get_root();
  {
    // Note: this is not a mutex
    CurrentCallableGuard _(program, this);
    func();
  }
  irpass::compile_inline_function(ir.get(), program->config, this,
                                  /*grad=*/false,
                                  /*verbose=*/program->config.print_ir,
                                  /*start_from_ast=*/true);

  taichi::lang::context = std::move(backup_context);
}

void Function::set_function_body(std::unique_ptr<IRNode> func_body) {
  ir = std::move(func_body);
  irpass::compile_inline_function(ir.get(), program->config, this,
                                  /*grad=*/false,
                                  /*verbose=*/program->config.print_ir,
                                  /*start_from_ast=*/false);
}

std::string Function::get_name() const {
  return func_key.get_full_name();
}

}  // namespace lang
}  // namespace taichi
