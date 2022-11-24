#include "taichi/ir/statements.h"

namespace taichi::lang {

Stmt *generate_mod(VecStatement *stmts, Stmt *x, int y) {
  if (bit::is_power_of_two(y)) {
    auto const_stmt = stmts->push_back<ConstStmt>(TypedConstant(y - 1));
    return stmts->push_back<BinaryOpStmt>(BinaryOpType::bit_and, x, const_stmt);
  }
  auto const_stmt = stmts->push_back<ConstStmt>(TypedConstant(y));
  return stmts->push_back<BinaryOpStmt>(BinaryOpType::mod, x, const_stmt);
}

Stmt *generate_div(VecStatement *stmts, Stmt *x, int y) {
  if (bit::is_power_of_two(y)) {
    auto const_stmt = stmts->push_back<ConstStmt>(
        TypedConstant(PrimitiveType::i32, bit::log2int(y)));
    return stmts->push_back<BinaryOpStmt>(BinaryOpType::bit_shr, x, const_stmt);
  }
  auto const_stmt = stmts->push_back<ConstStmt>(TypedConstant(y));
  return stmts->push_back<BinaryOpStmt>(BinaryOpType::div, x, const_stmt);
}

}  // namespace taichi::lang
