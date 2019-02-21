#include "structural_node.h"
#include "math.h"

TLANG_NAMESPACE_BEGIN

int SNode::counter = 0;

SNode &SNode::place(Matrix &mat) {
  for (auto &e : mat.entries) {
    this->place(e);
  }
  return *this;
}

SNode &SNode::place_new(ExpressionHandle &expr_) {
  TC_ASSERT(expr_.is<GlobalVariableExpression>());
  auto expr = expr_.cast<GlobalVariableExpression>();
  auto &child = insert_children(SNodeType::place);
  expr->snode = &child;
  name = expr->ident.name();

  child.dt = expr->dt;
  TC_WARN("Uncommenting this may lead to an RTE");
  return *this;
}

TLANG_NAMESPACE_END