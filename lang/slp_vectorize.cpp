#include "ir.h"

TLANG_NAMESPACE_BEGIN

class SLPVectorize : public IRVisitor {
 public:
  int vectorize;

  SLPVectorize() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    vectorize = 1;
  }

  void visit(Statement *stmt) {
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
    alloca->ret_type.width *= vectorize;
  }

  void visit(LocalLoadStmt *stmt) {
    if (vectorize == 1)
      return;
    stmt->ret_type.width *= vectorize;
    if (loop_var && stmt->ident == *loop_var) {
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
      stmt->insert_after(std::move(offsets));
      offsets_p->insert_after(std::move(add_op));
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
    for_stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }

  static void run(IRNode *node) {
    SLPVectorize inst;
    node->accept(&inst);
  }
};

namespace irpass {

void slp_vectorize(IRNode *root) {
  return SLPVectorize::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END