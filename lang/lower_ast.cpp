#include "ir.h"

TLANG_NAMESPACE_BEGIN

// Lower Expr tree to a bunch of binary/unary(binary/unary) statements
// Goal: eliminate Expression, and mutable local variables. Make AST SSA.
class LowerAST : public IRVisitor {
 public:
  LowerAST() {
    allow_undefined_visitor = true;
  }

  void load_if_ptr(VecStatement &flattened) {
    auto &ret = flattened.back();
    if (ret->is<GlobalPtrStmt>()) {
      flattened.push_back(
          std::make_unique<GlobalLoadStmt>(ret->as<GlobalPtrStmt>()));
    }
  }

  void visit(Block *stmt_list) {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(AllocaStmt *alloca) {
    // print("{} <- alloca {}", alloca->lhs.name(),
    // data_type_name(alloca->type));
  }

  void visit(FrontendIfStmt *stmt) override {
    VecStatement flattened;
    stmt->condition->flatten(flattened);
    auto new_if = std::make_unique<IfStmt>(flattened.back().get());
    new_if->true_statements = std::move(stmt->true_statements);
    new_if->false_statements = std::move(stmt->false_statements);
    flattened.push_back(std::move(new_if));
    stmt->parent->replace_with(stmt, flattened);
    throw IRModifiedException();
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(FrontendPrintStmt *stmt) {
    // expand rhs
    auto expr = stmt->expr;
    VecStatement flattened;
    expr->flatten(flattened);
    load_if_ptr(flattened);
    auto print = std::make_unique<PrintStmt>(flattened.back().get());
    flattened.push_back(std::move(print));
    stmt->parent->replace_with(stmt, flattened);
    throw IRModifiedException();
  }

  void visit(FrontendForStmt *stmt) {
    auto begin = stmt->begin;
    auto end = stmt->end;

    VecStatement flattened;

    begin->flatten(flattened);
    auto begin_stmt = flattened.back().get();

    end->flatten(flattened);
    auto end_stmt = flattened.back().get();

    flattened.push_back(std::make_unique<RangeForStmt>(
        stmt->loop_var_id, begin_stmt, end_stmt, std::move(stmt->body)));
    stmt->parent->replace_with(stmt, flattened);
    throw IRModifiedException();
  }

  void visit(RangeForStmt *for_stmt) {
    for_stmt->body->accept(this);
  }

  void visit(AssignStmt *assign) {
    // expand rhs
    auto expr = assign->rhs;
    VecStatement flattened;
    expr->flatten(flattened);
    auto rhs_stmt = flattened.back().get();
    if (assign->lhs.is<IdExpression>()) {  // local variable
      // emit local store stmt
      auto local_store = std::make_unique<LocalStoreStmt>(
          assign->lhs.cast<IdExpression>()->id, rhs_stmt);
      flattened.push_back(std::move(local_store));
    } else {  // global variable
      TC_ASSERT(assign->lhs.is<GlobalPtrExpression>());
      auto global_ptr = assign->lhs.cast<GlobalPtrExpression>();
      global_ptr->flatten(flattened);
      auto global_store =
          std::make_unique<GlobalStoreStmt>(flattened.back().get(), rhs_stmt);
      flattened.push_back(std::move(global_store));
    }
    assign->parent->replace_with(assign, flattened);
    throw IRModifiedException();
  }

  static void run(IRNode *node) {
    LowerAST inst;
    while (true) {
      bool modified = false;
      try {
        node->accept(&inst);
      } catch (IRModifiedException) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {

void lower(IRNode *root) {
  return LowerAST::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END