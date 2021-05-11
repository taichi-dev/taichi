#include "taichi/program/function.h"
#include "taichi/program/program.h"
#include "taichi/ir/transforms.h"

namespace taichi {
namespace lang {

namespace {
class CurrentFunctionGuard {
  std::variant<Kernel *, Function *> old_kernel_or_function;
  Program *program;

 public:
  CurrentFunctionGuard(Program *program, Function *func) : program(program) {
    old_kernel_or_function = program->current_kernel_or_function;
    program->current_kernel_or_function = func;
  }

  ~CurrentFunctionGuard() {
    program->current_kernel_or_function = old_kernel_or_function;
  }
};
}  // namespace

Function::Function(Program *program, const FunctionKey &func_key)
    : program(program), func_key(func_key) {
}

void Function::set_function_body(const std::function<void()> &func) {
  // Do not corrupt the context calling this function here
  auto backup_context = std::move(taichi::lang::context);

  taichi::lang::context = std::make_unique<FrontendContext>();
  ir = taichi::lang::context->get_root();
  {
    // Note: this is not a mutex
    CurrentFunctionGuard _(program, this);
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

int Function::insert_arg(DataType dt, bool is_external_array) {
  args.push_back(Arg{dt->get_compute_type(), is_external_array, /*size=*/0});
  return args.size() - 1;
}

int Function::insert_ret(DataType dt) {
  rets.push_back(Ret{dt->get_compute_type()});
  return rets.size() - 1;
}

}  // namespace lang
}  // namespace taichi
