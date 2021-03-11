#include "taichi/ir/ir.h"

#include "taichi/program/program.h"
#include "taichi/program/kernel.h"

namespace taichi {
namespace lang {

Kernel *IRNode::get_kernel() const {
  return const_cast<IRNode *>(this)->get_ir_root()->kernel;
}

CompileConfig &IRNode::get_config() const {
  return get_kernel()->program.config;
}

}  // namespace lang
}  // namespace taichi
