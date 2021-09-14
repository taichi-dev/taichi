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

void insert_snode_access_flag(SNodeAccessFlag v, const Expr &field) {
  dec.mem_access_opt.add_flag(field.snode(), v);
}

void reset_snode_access_flag() {
  dec.reset();
}

// Begin: legacy frontend constructs

For::For(const Expr &s, const Expr &e, const std::function<void(Expr)> &func) {
  auto i = Expr(std::make_shared<IdExpression>());
  auto stmt_unique = std::make_unique<FrontendForStmt>(i, s, e);
  auto stmt = stmt_unique.get();
  current_ast_builder().insert(std::move(stmt_unique));
  auto _ = current_ast_builder().create_scope(stmt->body);
  func(i);
}

For::For(const Expr &i,
         const Expr &s,
         const Expr &e,
         const std::function<void()> &func) {
  auto stmt_unique = std::make_unique<FrontendForStmt>(i, s, e);
  auto stmt = stmt_unique.get();
  current_ast_builder().insert(std::move(stmt_unique));
  auto _ = current_ast_builder().create_scope(stmt->body);
  func();
}

For::For(const ExprGroup &i,
         const Expr &global,
         const std::function<void()> &func) {
  auto stmt_unique = std::make_unique<FrontendForStmt>(i, global);
  auto stmt = stmt_unique.get();
  current_ast_builder().insert(std::move(stmt_unique));
  auto _ = current_ast_builder().create_scope(stmt->body);
  func();
}

// End: legacy frontend constructs

TLANG_NAMESPACE_END
