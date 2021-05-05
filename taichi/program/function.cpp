#include "taichi/program/function.h"
#include "taichi/program/program.h"
#include "taichi/ir/transforms.h"

namespace taichi {
namespace lang {

Function::Function(const std::string &funcid) : funcid(funcid) {
}

void Function::set_function_body(const std::function<void()> &func) {
  taichi::lang::context = std::make_unique<FrontendContext>();
  ir = taichi::lang::context->get_root();
  func();
  TI_INFO("AAAAA");
  irpass::print(ir.get());
}

}  // namespace lang
}  // namespace taichi
