#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

namespace {

class FixRootBlockKernel : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  explicit FixRootBlockKernel(Kernel *kernel) : kernel_(kernel) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Block *stmt_list) override {
    if (stmt_list->parent == nullptr) {
      stmt_list->kernel = kernel_;
    }
    // No need to visit leaves because we have found the root
  }

 private:
  Kernel *const kernel_;
};

}  // namespace

namespace irpass {

void fix_root_block_kernel(IRNode *root, Kernel *kernel) {
  FixRootBlockKernel f(kernel);
  root->accept(&f);
}

}  // namespace irpass

TLANG_NAMESPACE_END
