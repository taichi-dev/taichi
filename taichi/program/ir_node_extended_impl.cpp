#include "taichi/ir/ir.h"

#include "taichi/program/program.h"
#include "taichi/program/kernel.h"

// TODO(#2196): Part of the implementation of IRNode has to be placed here, in
// order to use files under "taichi/program". Ideally, we should:
// 1. Have a dedicated IR compile config that lives under "taichi/ir"
// 2. Just don't hook IRNode with a Kernel
namespace taichi {
namespace lang {

Kernel *IRNode::get_kernel() const {
  return const_cast<IRNode *>(this)->get_ir_root()->kernel;
}

Function *IRNode::get_if_func() const {
  if (const_cast<IRNode *>(this)->get_ir_root()->kernel == nullptr) {
    return const_cast<IRNode *>(this)->get_ir_root()->func;
  }
  return nullptr;
}

bool IRNode::has_real_func() const {
  if (auto func = get_if_func()) {
    return func->has_real_func;
  } else {
    auto kernel = get_kernel();
    if (kernel != nullptr)
      return kernel->has_real_func;
    else
      return false;
  }
}

void IRNode::set_has_real_func() {
  if (auto func = get_if_func()) {
    func->has_real_func = true;
  } else {
    auto kernel = get_kernel();
    if (kernel != nullptr)
      kernel->has_real_func = true;
  }
}

CompileConfig &IRNode::get_config() const {
  return get_kernel()->program->config;
}

}  // namespace lang
}  // namespace taichi
