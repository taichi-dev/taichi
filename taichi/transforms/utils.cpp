#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

Stmt *generate_mod_x_div_y(VecStatement *stmts, Stmt *num, int x, int y) {
  auto const_x = stmts->push_back<ConstStmt>(TypedConstant(x));
  auto mod_x = stmts->push_back<BinaryOpStmt>(BinaryOpType::mod, num, const_x);
  auto const_y = stmts->push_back<ConstStmt>(TypedConstant(y));
  return stmts->push_back<BinaryOpStmt>(BinaryOpType::div, mod_x, const_y);
}

TLANG_NAMESPACE_END
