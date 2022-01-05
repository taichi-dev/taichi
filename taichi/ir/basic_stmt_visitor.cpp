#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

BasicStmtVisitor::BasicStmtVisitor() {
  allow_undefined_visitor = true;
}

void BasicStmtVisitor::visit(Block *stmt_list) {
  std::vector<Stmt *> statements;
  // Make a copy in case the pass modifies the block itself
  for (auto &stmt : stmt_list->statements)
    statements.push_back(stmt.get());
  for (auto &stmt : statements)
    stmt->accept(this);
}

void BasicStmtVisitor::visit(IfStmt *if_stmt) {
  preprocess_container_stmt(if_stmt);
  if (if_stmt->true_statements)
    if_stmt->true_statements->accept(this);
  if (if_stmt->false_statements)
    if_stmt->false_statements->accept(this);
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

void BasicStmtVisitor::visit(MeshForStmt *for_stmt) {
  preprocess_container_stmt(for_stmt);
  for_stmt->body->accept(this);
}

void BasicStmtVisitor::visit(OffloadedStmt *stmt) {
  preprocess_container_stmt(stmt);
  stmt->all_blocks_accept(this);
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
