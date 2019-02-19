#include "ir.h"

TLANG_NAMESPACE_BEGIN

// Lower Expr tree to a bunch of binary/unary(binary/unary) statements
// Goal: eliminate Expression, and mutable local variables. Make AST SSA.
class LowerAST : public IRVisitor {
 public:
  LowerAST() {
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

  void visit(BinaryOpStmt *bin) {  // this will not appear here
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

  void visit(LocalLoadStmt *) {
  }

  void visit(LocalStoreStmt *) {
  }

  void visit(PrintStmt *stmt) {
  }

  void visit(FrontendPrintStmt *stmt) {
    // expand rhs
    auto expr = stmt->expr;
    VecStatement flattened;
    expr->flatten(flattened);
    auto print = std::make_unique<PrintStmt>(flattened.back().get());
    flattened.push_back(std::move(print));
    stmt->parent->replace_with(stmt, flattened);
    throw IRModifiedException();
  }

  void visit(ConstStmt *const_stmt) {  // this will not appear here
  }

  void visit(FrontendForStmt *for_stmt) {
    for_stmt->body->accept(this);
  }

  void visit(AssignStmt *assign) {
    // expand rhs
    auto expr = assign->rhs;
    VecStatement flattened;
    expr->flatten(flattened);
    if (true) {  // local variable
      // emit local store stmt
      auto local_store =
          std::make_unique<LocalStoreStmt>(assign->id, flattened.back().get());
      flattened.push_back(std::move(local_store));
    } else {  // global variable
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