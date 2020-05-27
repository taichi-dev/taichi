#include "taichi/ir/visitors.h"
#include "taichi/ir/frontend_ir.h"

TLANG_NAMESPACE_BEGIN

BasicStmtVisitor::BasicStmtVisitor() {
  allow_undefined_visitor = true;
}

void BasicStmtVisitor::visit(Block *stmt_list) {
  for (auto &stmt : stmt_list->statements) {
    stmt->accept(this);
  }
}

void BasicStmtVisitor::visit(IfStmt *if_stmt) {
  preprocess_container_stmt(if_stmt);
  if (if_stmt->true_statements)
    if_stmt->true_statements->accept(this);
  if (if_stmt->false_statements) {
    if_stmt->false_statements->accept(this);
  }
}

void BasicStmtVisitor::visit(WhileStmt *stmt) {
  preprocess_container_stmt(stmt);
  stmt->body->accept(this);
}

void BasicStmtVisitor::visit(RangeForStmt *for_stmt) {
  preprocess_container_stmt(for_stmt);
  for_stmt->body->accept(this);
}

void BasicStmtVisitor::visit(StructForStmt *for_stmt) {
  preprocess_container_stmt(for_stmt);
  for_stmt->body->accept(this);
}

void BasicStmtVisitor::visit(OffloadedStmt *stmt) {
  preprocess_container_stmt(stmt);
  if (stmt->body)
    stmt->body->accept(this);
}

void BasicStmtVisitor::visit(FuncBodyStmt *stmt) {
  preprocess_container_stmt(stmt);
  stmt->body->accept(this);
}

void BasicStmtVisitor::visit(FrontendWhileStmt *stmt) {
  preprocess_container_stmt(stmt);
  stmt->body->accept(this);
}

void BasicStmtVisitor::visit(FrontendForStmt *stmt) {
  preprocess_container_stmt(stmt);
  stmt->body->accept(this);
}

void BasicStmtVisitor::visit(FrontendIfStmt *stmt) {
  preprocess_container_stmt(stmt);
  if (stmt->true_statements)
    stmt->true_statements->accept(this);
  if (stmt->false_statements)
    stmt->false_statements->accept(this);
}

TLANG_NAMESPACE_END
