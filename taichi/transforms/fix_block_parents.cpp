#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

// Replace both usages and the statements themselves
class FixBlockParents : public IRVisitor {
 public:
  IRNode *node;
  Block *current_block;

  FixBlockParents(IRNode *node) : node(node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    current_block = nullptr;
  }

  void visit(Block *stmt_list) override {
    stmt_list->parent = current_block;

    current_block = stmt_list;
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }

    current_block = stmt_list->parent;
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->body)
      stmt->body->accept(this);
  }

  void run() {
    node->accept(this);
  }
};

namespace irpass {

void fix_block_parents(IRNode *root) {
  FixBlockParents transformer(root);
  transformer.run();
}

}  // namespace irpass

TLANG_NAMESPACE_END
