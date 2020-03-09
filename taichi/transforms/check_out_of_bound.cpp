#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

class CheckOutOfBound : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  CheckOutOfBound() : BasicStmtVisitor() {
  }

  static void run(IRNode *node) {
    ;
  }
};

namespace irpass {

void check_out_of_bound(IRNode *root) {
  return CheckOutOfBound::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
