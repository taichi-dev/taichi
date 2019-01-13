#include "node.h"
#include "expr.h"

TLANG_NAMESPACE_BEGIN

int Node::counter = 0;

int Node::group_size() const {
  return (int)members.size();
}

void Node::set_similar(taichi::Tlang::Expr expr) {
  set_lanes(expr->lanes);
  data_type = expr->data_type;
  binary_type = expr->binary_type;
  for (int i = 0; i < expr->lanes; i++) {
    active(i) = expr->active(i);
  }
}

TLANG_NAMESPACE_END
