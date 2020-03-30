#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

// Compare if two IRNodes are equivalent.
class IRNodeComparator : public IRVisitor {
 private:
  IRNode *other;

 public:
  bool same;

  IRNodeComparator(IRNode *other) : other(other) {
    allow_undefined_visitor = false;
    invoke_default_visitor = false;
    same = true;
  }

  static bool run(IRNode *root1, IRNode *root2) {
    IRNodeComparator comparator(root1);
    root2->accept(&comparator);
    return comparator.same;
  }
};

namespace irpass {
bool same_statements(IRNode *root1, IRNode *root2) {
  return IRNodeComparator::run(root1, root2);
}
}  // namespace irpass

TLANG_NAMESPACE_END
