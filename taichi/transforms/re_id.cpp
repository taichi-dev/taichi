#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/frontend_ir.h"

TLANG_NAMESPACE_BEGIN

// This pass manipulates the id of statements so that they are successive values
// starting from 0
class ReId : public IRVisitor {
 public:
  int id_counter;

  ReId(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    id_counter = 0;
    node->accept(this);
  }

  void re_id(Stmt *stmt) {
    stmt->id = id_counter++;
  }

  void visit(Stmt *stmt) {
    re_id(stmt);
  }

  void visit(Block *stmt_list) {  // block itself has no id
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(IfStmt *if_stmt) {
    re_id(if_stmt);
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(FrontendIfStmt *if_stmt) {
    re_id(if_stmt);
    if (if_stmt->true_statements)
      if (if_stmt->true_statements)
        if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) {
    re_id(stmt);
    stmt->body->accept(this);
  }

  void visit(FrontendWhileStmt *stmt) {
    re_id(stmt);
    stmt->body->accept(this);
  }

  void visit(FrontendForStmt *for_stmt) {
    re_id(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) {
    re_id(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) {
    re_id(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) {
    re_id(stmt);
    if (stmt->body)
      stmt->body->accept(this);
  }
};

namespace irpass {

void re_id(IRNode *root) {
  ReId instance(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
