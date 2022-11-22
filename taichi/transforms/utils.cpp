#include "taichi/ir/statements.h"

namespace taichi::lang {

Stmt *generate_mod(VecStatement *stmts, Stmt *x, int y) {
  auto const_stmt = stmts->push_back<ConstStmt>(TypedConstant(y));
  return stmts->push_back<BinaryOpStmt>(BinaryOpType::mod, x, const_stmt);
}

Stmt *generate_div(VecStatement *stmts, Stmt *x, int y) {
  auto const_stmt = stmts->push_back<ConstStmt>(TypedConstant(y));
  return stmts->push_back<BinaryOpStmt>(BinaryOpType::div, x, const_stmt);
}

}  // namespace taichi::lang
