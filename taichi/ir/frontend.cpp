// Frontend constructs

#include "taichi/ir/frontend.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

Expr global_new(Expr id_expr, DataType dt) {
  TI_ASSERT(id_expr.is<IdExpression>());
  auto ret = Expr(std::make_shared<GlobalVariableExpression>(
      dt, id_expr.cast<IdExpression>()->id));
  return ret;
}

Expr global_new(DataType dt, std::string name) {
  auto id_expr = std::make_shared<IdExpression>(name);
  return Expr::make<GlobalVariableExpression>(dt, id_expr->id);
}
TLANG_NAMESPACE_END
