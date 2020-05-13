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
  bool throw_ir_modified;
  bool modified;

  StatementReplace(IRNode *node,
                   std::function<bool(Stmt *)> filter,
                   std::function<std::unique_ptr<Stmt>()> generator,
                   bool throw_ir_modified)
      : node(node), filter(filter), generator(generator),
        throw_ir_modified(throw_ir_modified) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    modified = false;
  }

  void replace_if_necessary(Stmt *stmt) {
    if (filter(stmt)) {
//      std::cout << "begin replace\n";
      auto block = stmt->parent;
      auto new_stmt = generator();
      irpass::replace_all_usages_with(node, stmt, new_stmt.get());
      block->replace_with(stmt, std::move(new_stmt), false);
      if (throw_ir_modified)
        throw IRModified();
      else
        modified = true;
//      std::cout << "end replace\n";
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

  bool run() {
    while (true) {
      try {
        node->accept(this);
      } catch (IRModified) {
        continue;
      }
      break;
    }
    return modified;
  }
};

namespace irpass {

bool replace_statements_with(IRNode *root,
                             std::function<bool(Stmt *)> filter,
                             std::function<std::unique_ptr<Stmt>()> generator,
                             bool throw_ir_modified) {
  StatementReplace transformer(root, filter, generator, throw_ir_modified);
  return transformer.run();
}

}  // namespace irpass

TLANG_NAMESPACE_END
