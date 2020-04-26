#include "taichi/ir/ir.h"
#include "taichi/ir/state_machine.h"
#include <unordered_map>

TLANG_NAMESPACE_BEGIN

class VariableOptimize : public IRVisitor {
 private:
  std::unique_ptr<std::unordered_map<Stmt *, StateMachine>> state_machines;
  bool maybe_run;

 public:
  VariableOptimize() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    state_machines = std::make_unique<std::unordered_map<Stmt *, StateMachine>>();
    maybe_run = false;
  }

  StateMachine &get_state_machine(Stmt *stmt) {
    if (state_machines->find(stmt) == state_machines->end())
      state_machines->insert(std::make_pair(stmt, StateMachine(stmt)));
    return (*state_machines)[stmt];
  }

  static bool maybe_same_address(Stmt *var1, Stmt *var2) {
    return true;
  }

  void visit(AllocaStmt *stmt) override {
    if (state_machines->find(stmt) == state_machines->end())
      state_machines->insert(std::make_pair(stmt, StateMachine(stmt)));
    else
      (*state_machines)[stmt] = StateMachine(stmt);
  }

  void visit(Stmt *stmt) override {
    if (stmt->is_container_statement()) {
      TI_ERROR("Visitor for container stmt undefined.");
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    if (!stmt->dest->is<AllocaStmt>())
      return;
    if (maybe_run)
      get_state_machine(stmt->dest).maybe_atomic_op();
    else
      get_state_machine(stmt->dest).atomic_op(stmt);
    if (!stmt->dest->is<AllocaStmt>()) {
      for (auto &var : *state_machines) {
        if (var.first != stmt->dest && maybe_same_address(stmt->dest, var.first)) {
          var.second.maybe_atomic_op();
        }
      }
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    if (maybe_run)
      get_state_machine(stmt->ptr).maybe_store(stmt);
    else
      get_state_machine(stmt->ptr).store(stmt);
  }

  void visit(LocalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    TI_ASSERT(stmt->ptr[0].offset == 0);
    if (maybe_run)
      get_state_machine(stmt->ptr[0].var).maybe_load();
    else
      get_state_machine(stmt->ptr[0].var).load(stmt);
  }

  void visit(IfStmt *if_stmt) override {
    auto origin = std::move(state_machines);

    state_machines = std::make_unique<std::unordered_map<Stmt *, StateMachine>>();
    *state_machines = *origin;
    for (auto &it : *state_machines) {
      it.second.begin_if_or_loop();
    }
    if (if_stmt->true_statements) {
      if_stmt->true_statements->accept(this);
    }
    auto true_branch = std::move(state_machines);

    state_machines = std::make_unique<std::unordered_map<Stmt *, StateMachine>>();
    *state_machines = *origin;
    for (auto &it : *state_machines) {
      it.second.begin_if_or_loop();
    }
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
    auto false_branch = std::move(state_machines);

    state_machines = std::move(origin);
    for (auto &it : *state_machines) {
      it.second.merge_from_if((*true_branch)[it.first], (*false_branch)[it.first]);
    }

    for (auto &it : *true_branch) {
      if (!it.first->is<AllocaStmt>() && state_machines->find(it.first) == state_machines->end())
        state_machines->insert(it);
    }
    for (auto &it : *false_branch) {
      if (!it.first->is<AllocaStmt>() && state_machines->find(it.first) == state_machines->end())
        state_machines->insert(it);
    }
  }

  void visit_loop(Block *body, const std::vector<Stmt *> &loop_vars) {
    if (maybe_run) {
      body->accept(this);
      return;
    }

    auto origin = std::move(state_machines);

    state_machines = std::make_unique<std::unordered_map<Stmt *, StateMachine>>();
    *state_machines = *origin;
    for (auto &it : *state_machines) {
      it.second.begin_if_or_loop();
    }
    for (auto &loop_var : loop_vars) {
      get_state_machine(loop_var).mark_as_loop_var();
    }
    maybe_run = true;
    body->accept(this);
    maybe_run = false;
    for (auto &it : *origin) {
      it.second.merge_from_loop((*state_machines)[it.first]);
    }
    for (auto &it : *state_machines) {
      if (!it.first->is<AllocaStmt>() && state_machines->find(it.first) == state_machines->end())
        origin->insert(it);
    }
    body->accept(this);
    state_machines = std::move(origin);
  }

  void visit(Block *block) override {
    for (auto &stmt : block->statements) {
      stmt->accept(this);
    }
  }

  void visit(WhileStmt *stmt) override {
    TI_ASSERT(stmt->mask == nullptr);
    visit_loop(stmt->body.get(), {});
  }

  void visit(RangeForStmt *stmt) override {
    visit_loop(stmt->body.get(), {stmt->loop_var});
  }

  void visit(StructForStmt *stmt) override {
    visit_loop(stmt->body.get(), stmt->loop_vars);
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->body) {
      for (auto &it : *state_machines) {
        it.second.mark_new_offload();
      }
      stmt->body->accept(this);
    }
  }

  void clear() {
    for (auto &it : *state_machines)
      it.second.finalize();
    state_machines->clear();
  }

  static void run(IRNode *node) {
    StateMachine::rebuild_atomics_usage(node);
    VariableOptimize optimizer;
    while (true) {
      bool modified = false;
      try {
        node->accept(&optimizer);
        optimizer.clear();
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {
void variable_optimization(IRNode *root) {
  VariableOptimize::run(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
