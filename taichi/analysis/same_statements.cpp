#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

#define DEFINE_TYPECHECK(Type)        \
  if (!other_node->is<Type>()) {      \
    same = false;                     \
    return;                           \
  }                                   \
  auto other = other_node->as<Type>()

// Compare if two IRNodes are equivalent.
class IRNodeComparator : public IRVisitor {
 private:
  IRNode *other_node;

 public:
  bool same;

  IRNodeComparator(IRNode *other_node) : other_node(other_node) {
    same = true;
  }

  void visit(Block *stmt_list) override {
    DEFINE_TYPECHECK(Block);
    if (stmt_list->statements.size() != other->statements.size()) {
      same = false;
      return;
    }
    for (int i = 0; i < (int)stmt_list->statements.size(); i++) {
      other_node = other->statements[i].get();
      stmt_list->statements[i]->accept(this);
      if (!same)
        break;
    }
    other_node = other;
  }

  static bool run(IRNode *root1, IRNode *root2) {
    IRNodeComparator comparator(root2);
    root1->accept(&comparator);
    return comparator.same;
  }
};

namespace irpass {
bool same_statements(IRNode *root1, IRNode *root2) {
  return IRNodeComparator::run(root1, root2);
}
}  // namespace irpass

TLANG_NAMESPACE_END
