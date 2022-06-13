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
  context = std::make_unique<FrontendContext>(program->config.arch);
  ir = context->get_root();
  {
    // Note: this is not a mutex
    CurrentCallableGuard _(program, this);
    func();
  }
  irpass::compile_function(ir.get(), program->config, this,
                           /*autodiff_mode=*/AutodiffMode::kNone,
                           /*verbose=*/program->config.print_ir,
                           /*start_from_ast=*/true);
}

void Function::set_function_body(std::unique_ptr<IRNode> func_body) {
  ir = std::move(func_body);
  irpass::compile_function(ir.get(), program->config, this,
                           /*autodiff_mode=*/AutodiffMode::kNone,
                           /*verbose=*/program->config.print_ir,
                           /*start_from_ast=*/false);
}

std::string Function::get_name() const {
  return func_key.get_full_name();
}

}  // namespace lang
}  // namespace taichi
