#pragma once
#include "statements.h"

TLANG_NAMESPACE_BEGIN

// Visits all non-containing statements
class BasicStmtVisitor : public IRVisitor {
 private:
  StructForStmt *current_struct_for;

 public:
  BasicStmtVisitor();

  void visit(Block *stmt_list) override;

  void visit(IfStmt *if_stmt) override;

  void visit(WhileStmt *stmt) override;

  void visit(RangeForStmt *for_stmt) override;

  void visit(StructForStmt *for_stmt) override;

  void visit(OffloadedStmt *stmt) override;
};

TLANG_NAMESPACE_END
