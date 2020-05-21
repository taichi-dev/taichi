#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

// Replace both usages and the statements themselves
class StatementReplace : public IRVisitor {
 public:
  IRNode *node;
  std::function<bool(Stmt *)> filter;
  std::function<std::unique_ptr<Stmt>()> generator;

  StatementReplace(IRNode *node,
                   std::function<bool(Stmt *)> filter,
                   std::function<std::unique_ptr<Stmt>()> generator)
      : node(node), filter(filter), generator(generator) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void replace_if_necessary(Stmt *stmt) {
    if (filter(stmt)) {
      auto block = stmt->parent;
      auto new_stmt = generator();
      irpass::replace_all_usages_with(node, stmt, new_stmt.get());
      block->replace_with(stmt, std::move(new_stmt), false);
    }
  }

  void visit(Block *stmt_list) override {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(IfStmt *if_stmt) override {
    replace_if_necessary(if_stmt);
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) override {
    replace_if_necessary(stmt);
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    replace_if_necessary(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    replace_if_necessary(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(Stmt *stmt) override {
    replace_if_necessary(stmt);
  }

  void run() {
    node->accept(this);
  }
};

namespace irpass {

void replace_statements_with(IRNode *root,
                             std::function<bool(Stmt *)> filter,
                             std::function<std::unique_ptr<Stmt>()> generator) {
  StatementReplace transformer(root, filter, generator);
  transformer.run();
}

}  // namespace irpass

TLANG_NAMESPACE_END
