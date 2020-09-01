#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {

void fix_root_block_kernel(IRNode *root, Kernel *kernel) {
  TI_ASSERT(root->get_parent() == nullptr);
  root->kernel = kernel;
}

}  // namespace irpass

TLANG_NAMESPACE_END
