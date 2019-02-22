#include "ir.h"

TLANG_NAMESPACE_BEGIN

class StatementReplace : public IRVisitor {
 public:
  Stmt *old_stmt, *new_stmt;

  StatementReplace() {
    allow_undefined_visitor = true;
  }

  void visit(IfStmt *if_stmt) {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(Block *stmt_list) {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(RangeForStmt *stmt) {
    auto block = stmt->parent;
    TC_ASSERT(block->local_variables.find(stmt->loop_var) ==
              block->local_variables.end());
    mark_as_if_const(stmt->begin, VectorType(1, DataType::i32));
    mark_as_if_const(stmt->end, VectorType(1, DataType::i32));
    block->local_variables.insert(
        std::make_pair(stmt->loop_var, VectorType(1, DataType::i32)));
    stmt->body->accept(this);
  }

  static void run(IRNode *node, IRNode *old_stmt, IRNode *new_stmt) {
    StatementReplace inst(old_stmt, new_stmt);
    node->accept(&inst);
  }
};

namespace irpass {

void typecheck(IRNode *root) {
  void replace_all_usages_with(IRNode * root, IRNode * old_stmt,
                               IRNode * new_stmt) {
    StatementReplace::run(root, old_stmt, new_stmt);
  }
}

}  // namespace irpass

TLANG_NAMESPACE_END
