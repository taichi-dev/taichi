#include "taichi/ir/ir.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// Optimize one alloca
class AllocaOptimize : public IRVisitor {
 private:
  AllocaStmt *alloca_stmt;

 public:
  bool stored;  // Is this alloca ever stored?
  bool loaded;  // Is this alloca ever loaded?

  LocalStoreStmt *last_store;

  // last_store_valid: Can we do store-forwarding?
  // When the last store is conditional, last_store_invalid is false.
  bool last_store_valid;

  // last_store_loaded: Is the last store ever loaded? If not, eliminate it.
  // If stored is false, last_store_loaded means if the alloca is ever loaded.
  bool last_store_loaded;

  AtomicOpStmt *last_atomic;

  // last_atomic_eliminable: Can we eliminate last_atomic if no statements
  // include it as an operand?
  bool last_atomic_eliminable;

  // Are we inside a loop which is inside the alloca's scope?
  // inside_loop == 0: No
  // inside_loop == 1: Yes
  // inside_loop == 2: Yes, but we've already checked that there are no
  //                   local stores in the loop and before the loop
  //                   (so that we can optimize local loads to const [0]).
  int inside_loop;

  explicit AllocaOptimize(AllocaStmt *alloca_stmt)
      : alloca_stmt(alloca_stmt),
        stored(false),
        loaded(false),
        last_store(nullptr),
        last_store_valid(false),
        last_store_loaded(false),
        last_atomic(nullptr),
        last_atomic_eliminable(false),
        inside_loop(0) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    if (stmt->is_container_statement()) {
      TI_ERROR("Visitor for container stmt undefined.");
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    if (stmt->dest != alloca_stmt)
      return;
    stored = true;
    loaded = true;
    last_store = nullptr;
    last_store_valid = false;
    last_store_loaded = false;
    last_atomic = stmt;
    last_atomic_eliminable = true;
  }

  void visit(LocalStoreStmt *stmt) override {
    if (stmt->ptr != alloca_stmt)
      return;
    stored = true;
    last_store = stmt;
    last_store_valid = true;
    last_store_loaded = false;
    last_atomic = nullptr;
    last_atomic_eliminable = false;
  }

  void visit(LocalLoadStmt *stmt) override {
    bool regular = true;
    for (int l = 0; l < stmt->width(); l++) {
      if (stmt->ptr[l].offset != l || stmt->ptr[l].var != alloca_stmt) {
        regular = false;
      }
      if (stmt->ptr[l].var == alloca_stmt) {
        loaded = true;
        if (last_store)
          last_store_loaded = true;
        if (last_atomic)
          last_atomic_eliminable = false;
      }
    }
    if (!regular)
      return;
    if (!stored && inside_loop != 1) {
      auto zero = stmt->insert_after_me(Stmt::make<ConstStmt>(
          LaneAttribute<TypedConstant>(alloca_stmt->ret_type.data_type)));
      zero->repeat(stmt->width());
      int current_stmt_id = stmt->parent->locate(stmt);
      stmt->replace_with(zero);
      stmt->parent->erase(current_stmt_id);
      throw IRModified();
    }
    if (last_store_valid) {
      // store-forwarding
      stmt->replace_with(last_store->data);
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  void visit(IfStmt *if_stmt) override {
    TI_ASSERT(if_stmt->true_mask == nullptr);
    TI_ASSERT(if_stmt->false_mask == nullptr);

    // Create two new instances for IfStmt
    AllocaOptimize true_branch = *this;
    AllocaOptimize false_branch = *this;
    if (if_stmt->true_statements) {
      if_stmt->true_statements->accept(&true_branch);
    }
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(&false_branch);
    }

    stored = true_branch.stored || false_branch.stored;
    loaded = true_branch.loaded || false_branch.loaded;

    if (!stored) {
      // do nothing to last_store
    } else if (true_branch.last_store_valid && false_branch.last_store_valid &&
               true_branch.last_store == false_branch.last_store) {
      TI_ASSERT(true_branch.last_store != nullptr);
      last_store_valid = true;
      if (last_store == true_branch.last_store) {
        last_store_loaded = last_store_loaded ||
                            true_branch.last_store_loaded ||
                            false_branch.last_store_loaded;
      } else {
        last_store = true_branch.last_store;
        last_store_loaded =
            true_branch.last_store_loaded || false_branch.last_store_loaded;
      }
    } else {
      last_store_valid = false;
      // Since it's invalid, we only care if we can eliminate the last store.
      if (true_branch.last_store == last_store &&
          false_branch.last_store == last_store) {
        // The last store didn't change.
        last_store_loaded = last_store_loaded ||
                            true_branch.last_store_loaded ||
                            false_branch.last_store_loaded;
      } else {
        // The last store changed, so we can't eliminate last_store.
        bool true_eliminable = true_branch.last_store != last_store &&
                               true_branch.last_store != nullptr &&
                               !true_branch.last_store_loaded;
        bool false_eliminable = false_branch.last_store != last_store &&
                                false_branch.last_store != nullptr &&
                                !false_branch.last_store_loaded;
        if (true_eliminable) {
          last_store = true_branch.last_store;
          last_store_loaded = false;
        } else if (false_eliminable) {
          last_store = false_branch.last_store;
          last_store_loaded = false;
        } else {
          // Neither branch provides a eliminable local store.
          last_store = nullptr;
          last_store_loaded = false;
        }
      }
    }

    if (true_branch.last_atomic == last_atomic &&
        false_branch.last_atomic == last_atomic) {
      // The last AtomicOpStmt didn't change.
      last_atomic_eliminable = last_atomic_eliminable &&
                               true_branch.last_atomic_eliminable &&
                               false_branch.last_atomic_eliminable;
    } else {
      // The last AtomicOpStmt changed, so we can't eliminate last_atomic.
      bool true_eliminable = true_branch.last_atomic != last_atomic &&
                             true_branch.last_atomic != nullptr &&
                             true_branch.last_atomic_eliminable;
      bool false_eliminable = false_branch.last_atomic != last_atomic &&
                              false_branch.last_atomic != nullptr &&
                              false_branch.last_atomic_eliminable;
      if (true_eliminable) {
        last_atomic = true_branch.last_atomic;
        last_atomic_eliminable = true;
      } else if (false_eliminable) {
        last_atomic = false_branch.last_atomic;
        last_atomic_eliminable = true;
      } else {
        // Neither branch provides a eliminable AtomicOpStmt.
        last_atomic = nullptr;
        last_atomic_eliminable = false;
      }
    }
  }

  void visit(Block *block) override {
    TI_ASSERT(block != alloca_stmt->parent);
    for (auto &stmt : block->statements) {
      stmt->accept(this);
    }
  }

  void visit_loop(Block *body, bool is_loop_var) {
    TI_ASSERT(body);
    if (is_loop_var) {
      TI_ASSERT(inside_loop == 0);  // no nested loops with the same alloca
    }
    AllocaOptimize loop(alloca_stmt);
    loop.inside_loop = 1;
    if (inside_loop == 2) {
      // Already checked that there are no stores inside.
      loop.inside_loop = 2;
    }
    body->accept(&loop);

    stored = stored || loop.stored;
    loaded = loaded || loop.loaded;

    if (is_loop_var) {
      // Don't do any optimization about the loop var.
      stored = true;
      loaded = true;
      last_store = nullptr;
      last_store_valid = false;
      last_store_loaded = false;
      last_atomic = nullptr;
      last_atomic_eliminable = false;
    } else if (!loop.stored) {
      // Since the loop does not store the alloca,
      // we can do store-forwarding.
      if (loop.loaded && inside_loop != 2 &&
          ((!stored && inside_loop != 1) || last_store_valid)) {
        loop = *this;
        loop.inside_loop = 2;
        body->accept(&loop);
      }
      // And the status about the last store should not be changed.
    } else {
      // The loop stores the alloca, and it must be invalid now
      // as we don't know if the loop is fully executed.
      last_store = loop.last_store;
      last_store_valid = false;
      last_atomic = loop.last_atomic;
      if (loop.loaded) {
        // The loop loads the alloca, so we cannot eliminate any stores
        // or AtomicOpStmts in the loop.
        last_store_loaded = true;
        last_atomic_eliminable = false;
      } else {
        // The loop stores the alloca but never loads it.
        last_store_loaded = false;
        last_atomic_eliminable = true;
      }
    }
  }

  void visit(WhileStmt *stmt) override {
    TI_ASSERT(stmt->mask == nullptr);
    visit_loop(stmt->body.get(), false);
  }

  void visit(RangeForStmt *stmt) override {
    visit_loop(stmt->body.get(), stmt->loop_var == alloca_stmt);
  }

  void visit(StructForStmt *stmt) override {
    bool is_loop_var = false;
    for (auto &loop_var : stmt->loop_vars) {
      if (loop_var == alloca_stmt) {
        is_loop_var = true;
        break;
      }
    }
    visit_loop(stmt->body.get(), is_loop_var);
  }

  void run() {
    Block *block = alloca_stmt->parent;
    TI_ASSERT(block);
    int location = block->locate(alloca_stmt);
    TI_ASSERT(location != -1);
    for (int i = location + 1; i < (int)block->size(); i++) {
      block->statements[i]->accept(this);
    }
    if (last_store && !last_store_loaded) {
      // The last store is never loaded.
      // last_store_valid == false means that it's in an IfStmt.
      // Eliminate the last store.
      last_store->parent->erase(last_store);
      throw IRModified();
    }
    if (last_atomic && last_atomic_eliminable) {
      // The last AtomicOpStmt is never loaded.
      // last_atomic_valid == false means that it's in an IfStmt.
      if (irpass::analysis::gather_statements(
              block,
              [&](Stmt *stmt) { return stmt->have_operand(last_atomic); })
              .empty()) {
        // The last AtomicOpStmt is never used.
        // Eliminate the last AtomicOpStmt.
        last_atomic->parent->erase(last_atomic);
        throw IRModified();
      }
    }
    if (!stored && !loaded) {
      // Never stored and never loaded.
      if (irpass::analysis::gather_statements(
              block,
              [&](Stmt *stmt) { return stmt->have_operand(alloca_stmt); })
              .empty()) {
        // Eliminate this alloca.
        block->erase(alloca_stmt);
        throw IRModified();
      }
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
