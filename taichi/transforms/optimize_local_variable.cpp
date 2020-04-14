#include "taichi/ir/ir.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// Optimize one alloca
class AllocaOptimize : public BasicStmtVisitor {
 private:
  AllocaStmt *alloca;

 public:
  using BasicStmtVisitor::visit;

  explicit AllocaOptimize(AllocaStmt *alloca) : alloca(alloca) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void run() {
    Block *block = alloca->parent;
    TI_ASSERT(block);
    int location = block->locate(alloca);
    TI_ASSERT(location != -1);
    for (int i = location + 1; i < (int)block->size(); i++) {
      block->statements[i]->accept(this);
    }
  }


};

class AllocaFindAndOptimize : public BasicStmtVisitor {
 private:
  std::unordered_set<int> visited;

 public:
  using BasicStmtVisitor::visit;

  AllocaFindAndOptimize() : visited() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  bool is_done(Stmt *stmt) {
    return visited.find(stmt->instance_id) != visited.end();
  }

  void set_done(Stmt *stmt) {
    visited.insert(stmt->instance_id);
  }

  void visit(AllocaStmt *alloca_stmt) override {
    if (is_done(alloca_stmt))
      return;
    AllocaOptimize optimizer(alloca_stmt);
    optimizer.run();
    set_done(alloca_stmt);
  }

  static void run(IRNode *node) {
    AllocaFindAndOptimize find_and_optimizer;
    while (true) {
      bool modified = false;
      try {
        node->accept(&find_and_optimizer);
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {
void optimize_local_variable(IRNode *root) {
  AllocaFindAndOptimize::run(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
