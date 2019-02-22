#include "ir.h"

TLANG_NAMESPACE_BEGIN

// Lower Expr tree to a bunch of binary/unary(binary/unary) statements
// Goal: eliminate Expression, and mutable local variables. Make AST SSA.
class LoopVectorize : public IRVisitor {
 public:
  int vectorize;

  LoopVectorize() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    vectorize = 1;
  }

  void visit(Statement *stmt) {
    stmt->ret_type.width *= vectorize;
  }

  void visit(Block *stmt_list) {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(AllocaStmt *alloca) {
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(RangeForStmt *for_stmt) {
    auto old_vectorize = for_stmt->vectorize;
    vectorize = for_stmt->vectorize;
    for_stmt->body->accept(this);
    for_stmt->vectorize = old_vectorize;
  }

  static void run(IRNode *node) {
    LoopVectorize inst;
    node->accept(&inst);
  }
};

namespace irpass {

void loop_vectorize(IRNode *root) {
  return LoopVectorize::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END