#include "taichi/ir/ir.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// Optimize one alloca
class AllocaOptimize : public BasicStmtVisitor {
 private:
  AllocaStmt *alloca;
  bool stored; // Is this alloca ever stored?
  LocalStoreStmt *last_store;
  bool last_store_valid;
  bool last_store_loaded; // Is the last store ever loaded?
  AtomicOpStmt *last_atomic;
  bool last_atomic_valid;
  bool last_atomic_eliminable;

 public:
  using BasicStmtVisitor::visit;

  explicit AllocaOptimize(AllocaStmt *alloca)
      : alloca(alloca),
        stored(false),
        last_store(nullptr),
        last_store_valid(false),
        last_store_loaded(false),
        last_atomic(nullptr),
        last_atomic_valid(false),
        last_atomic_eliminable(false) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    // temporarily being overcautious here
    stored = true;
    last_store_valid = false;
    last_atomic_valid = false;
  }

  void visit(AtomicOpStmt *stmt) override {
    if (stmt->dest != alloca)
      return;
    stored = true;
    last_store_valid = false;
    last_atomic = stmt;
    last_atomic_valid = true;
    last_atomic_eliminable = true;
  }

  void visit(LocalStoreStmt *stmt) override {
    if (stmt->ptr != alloca)
      return;
    stored = true;
    last_store = stmt;
    last_store_valid = true;
    last_store_loaded = false;
    last_atomic_valid = false;
  }

  void visit(LocalLoadStmt *stmt) override {
    bool regular = true;
    for (int l = 0; l < stmt->width(); l++) {
      if (stmt->ptr[l].offset != l || stmt->ptr[l].var != alloca) {
        regular = false;
      }
      if (stmt->ptr[l].var == alloca) {
        last_store_loaded = true;
        if (last_atomic_valid)
          last_atomic_eliminable = false;
      }
    }
    if (!regular)
      return;
    if (!stored) {
      auto zero = stmt->insert_after_me(Stmt::make<ConstStmt>(
          LaneAttribute<TypedConstant>(alloca->ret_type.data_type)));
      zero->repeat(stmt->width());
      int current_stmt_id = stmt->parent->locate(stmt);
      stmt->replace_with(zero);
      stmt->parent->erase(current_stmt_id);
      throw IRModified();
    }
    //TODO
  }

  void run() {
    Block *block = alloca->parent;
    TI_ASSERT(block);
    int location = block->locate(alloca);
    TI_ASSERT(location != -1);
    for (int i = location + 1; i < (int)block->size(); i++) {
      block->statements[i]->accept(this);
    }
    if (last_store_valid && !last_store_loaded) {
      // The last store is never loaded.
      // Eliminate the last store.
      TI_ASSERT(last_store);
      last_store->parent->erase(last_store);
      throw IRModified();
    }
    if (last_atomic_valid && last_atomic_eliminable) {
      // The last AtomicOpStmt is never loaded.
      TI_ASSERT(last_atomic);
      if (irpass::analysis::gather_statements(
          block, [&](Stmt *stmt) {
            return stmt->have_operand(last_atomic);
          }).empty()) {
        // The last AtomicOpStmt is never used.
        // Eliminate the last AtomicOpStmt.
        last_atomic->parent->erase(last_atomic);
        throw IRModified();
      }
    }
    if (!stored && !last_store_loaded) {
      // Never stored and never loaded.
      // Eliminate this alloca.
      block->erase(location);
      throw IRModified();
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
  std::cout << "before optimize_local_variable:\n";
  print(root);
  AllocaFindAndOptimize::run(root);
  std::cout << "after optimize_local_variable:\n";
  print(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
