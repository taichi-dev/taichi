#include "snode.h"
#include "math.h"

TLANG_NAMESPACE_BEGIN

int SNode::counter = 0;

/*
SNode &SNode::place(Matrix &mat) {
  for (auto &e : mat.entries) {
    this->place(e);
  }
  return *this;
}
*/

SNode &SNode::place(Expr &expr_) {
  TC_ASSERT(expr_.is<GlobalVariableExpression>());
  auto expr = expr_.cast<GlobalVariableExpression>();
  auto &child = insert_children(SNodeType::place);
  expr->snode = &child;
  name = expr->ident.name();
  if (expr->has_ambient) {
    expr->snode->has_ambient = true;
    expr->snode->ambient_val = expr->ambient_value;
  }
  child.dt = expr->dt;
  return *this;
}

TLANG_NAMESPACE_END