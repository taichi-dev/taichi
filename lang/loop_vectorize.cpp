#include "ir.h"

TLANG_NAMESPACE_BEGIN

// Lower Expr tree to a bunch of binary/unary(binary/unary) statements
// Goal: eliminate Expression, and mutable local variables. Make AST SSA.
class LoopVectorize : public IRVisitor {
 public:
  int vectorize;
  Ident *loop_var;

  LoopVectorize() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    loop_var = nullptr;
    vectorize = 1;
  }

  void visit(Statement *stmt) {
    stmt->ret_type.width *= vectorize;
  }

  void visit(Block *stmt_list) {
    std::vector<Stmt *> statements;
    for (auto &stmt : stmt_list->statements) {
      statements.push_back(stmt.get());
    }
    for (auto stmt : statements) {
      stmt->accept(this);
    }
  }

  void visit(AllocaStmt *alloca) {
  }

  void visit(LocalLoadStmt *stmt) {
    if (vectorize == 1)
      return;
    TC_ASSERT(stmt->ident.size() == 1);
    stmt->ret_type.width *= vectorize;
    stmt->ident.repeat(vectorize);
    if (loop_var && stmt->ident[0] == *loop_var) {
      // insert_before
      auto offsets = std::make_unique<ConstStmt>(0);
      offsets->repeat(vectorize);
      for (int i = 0; i < vectorize; i++) {
        offsets->value[i] = i;
      }
      auto add_op =
          std::make_unique<BinaryOpStmt>(BinaryType::add, stmt, offsets.get());
      irpass::typecheck(add_op.get());
      auto offsets_p = offsets.get();
      stmt->replace_with(add_op.get());
      TC_TAG;
      stmt->insert_after(std::move(offsets));
      TC_TAG;
      offsets_p->insert_after(std::move(add_op));
      TC_TAG;
    }
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
    loop_var = &for_stmt->loop_var;
    for_stmt->body->accept(this);
    loop_var = nullptr;
    vectorize = old_vectorize;
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