#include "visitors.h"

TLANG_NAMESPACE_BEGIN

BasicStmtVisitor::BasicStmtVisitor() {
  current_struct_for = nullptr;
  allow_undefined_visitor = true;
}

void BasicStmtVisitor::visit(Block *stmt_list) {
  auto backup_block = current_block;
  current_block = stmt_list;
  for (auto &stmt : stmt_list->statements) {
    stmt->accept(this);
  }
  current_block = backup_block;
}

void BasicStmtVisitor::visit(IfStmt *if_stmt) {
  if (if_stmt->true_statements)
    if_stmt->true_statements->accept(this);
  if (if_stmt->false_statements) {
    if_stmt->false_statements->accept(this);
  }
}

void BasicStmtVisitor::visit(WhileStmt *stmt) {
  stmt->body->accept(this);
}

void BasicStmtVisitor::visit(RangeForStmt *for_stmt) {
  for_stmt->body->accept(this);
}

void BasicStmtVisitor::visit(StructForStmt *for_stmt) {
  current_struct_for = for_stmt;
  for_stmt->body->accept(this);
  current_struct_for = nullptr;
}

void BasicStmtVisitor::visit(OffloadedStmt *stmt) {
  if (stmt->body)
    stmt->body->accept(this);
}

TLANG_NAMESPACE_END
