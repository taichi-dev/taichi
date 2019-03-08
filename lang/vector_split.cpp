#include "ir.h"

TLANG_NAMESPACE_BEGIN

// Goal: eliminate vectors that are longer than physical vector width (e.g. 8 on
// AVX2)
class VectorSplit : public IRVisitor {
 public:
  VectorSplit() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Statement *stmt) {
  }

  void visit(ConstStmt *stmt) {
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

  void visit(GlobalPtrStmt *ptr) {
  }

  void visit(AllocaStmt *alloca) {
  }

  void visit(LocalLoadStmt *stmt) {
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
    for_stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }

  static void run(IRNode *node) {
    VectorSplit inst;
    node->accept(&inst);
  }
};

namespace irpass {

void vector_split(IRNode *root) {
  return VectorSplit::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END