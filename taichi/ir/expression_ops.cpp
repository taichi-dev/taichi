#include "taichi/ir/operation_impl.h"
#include "taichi/ir/frontend_ir.h"

namespace taichi {
namespace lang {

#define TI_EXPRESSION_IMPLEMENTATION
#include "taichi/ir/expression_ops.h"

}  // namespace lang
}  // namespace taichi
