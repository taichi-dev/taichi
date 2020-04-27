#include "taichi/ir/ir.h"
#include "taichi/ir/state_machine.h"
#include <unordered_map>

TLANG_NAMESPACE_BEGIN

class VariableOptimize : public IRVisitor {
 protected:
  std::unique_ptr<std::unordered_map<Stmt *, StateMachine>> state_machines;
  bool maybe_run;

 public:
  VariableOptimize() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    state_machines =
        std::make_unique<std::unordered_map<Stmt *, StateMachine>>();
    maybe_run = false;
  }

  virtual StateMachine &get_state_machine(Stmt *stmt) {
    if (state_machines->find(stmt) == state_machines->end())
      state_machines->insert(std::make_pair(stmt, StateMachine(stmt)));
    return (*state_machines)[stmt];
  }

  virtual void modify_all_state_machines(void (StateMachine::*func)()) {
    for (auto &it : *state_machines) {
      (it.second.*func)();
    }
  }

  virtual void clear() {
    state_machines->clear();
  }

  static bool maybe_same_address(Stmt *var1, Stmt *var2) {
    return true;
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
        if (var.first != stmt->dest &&
            maybe_same_address(stmt->dest, var.first)) {
          var.second.maybe_atomic_op();
        }
      }
    }
  }

  void visit(WhileControlStmt *stmt) override {
    if (!maybe_run) {
      modify_all_state_machines(&StateMachine::continue_or_break);
    }
  }

  void visit(ContinueStmt *stmt) override {
    if (!maybe_run) {
      modify_all_state_machines(&StateMachine::continue_or_break);
    }
  }

  void visit(IfStmt *if_stmt) override {
    auto origin = std::move(state_machines);

    state_machines =
        std::make_unique<std::unordered_map<Stmt *, StateMachine>>();
    *state_machines = *origin;
    for (auto &it : *state_machines) {
      it.second.begin_if_or_loop();
    }
    if (if_stmt->true_statements) {
      if_stmt->true_statements->accept(this);
    }
    auto true_branch = std::move(state_machines);

    state_machines =
        std::make_unique<std::unordered_map<Stmt *, StateMachine>>();
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
      it.second.merge_from_if((*true_branch)[it.first],
                              (*false_branch)[it.first]);
    }

    for (auto &it : *true_branch) {
      if (!it.first->is<AllocaStmt>() &&
          state_machines->find(it.first) == state_machines->end())
        state_machines->insert(it);
    }
    for (auto &it : *false_branch) {
      if (!it.first->is<AllocaStmt>() &&
          state_machines->find(it.first) == state_machines->end())
        state_machines->insert(it);
    }
  }

  virtual void visit_loop(Block *body, const std::vector<Stmt *> &loop_vars) {
    if (maybe_run) {
      body->accept(this);
      return;
    }

    auto origin = std::move(state_machines);

    state_machines =
        std::make_unique<std::unordered_map<Stmt *, StateMachine>>();
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
    body->accept(this);
    for (auto &it : *origin) {
      it.second.merge_from_loop((*state_machines)[it.first]);
    }
    for (auto &it : *state_machines) {
      if (!it.first->is<AllocaStmt>() &&
          state_machines->find(it.first) == state_machines->end())
        origin->insert(it);
    }
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
      modify_all_state_machines(&StateMachine::begin_offload);
      stmt->body->accept(this);
    }
  }

  void finalize() {
    modify_all_state_machines(&StateMachine::finalize);
  }

  void run(IRNode *node) {
    StateMachine::rebuild_atomics_usage(node);
    while (true) {
      bool modified = false;
      try {
        clear();
        node->accept(this);
        finalize();
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

class AllocaOptimize : public VariableOptimize {
 private:
  std::unordered_map<Block *, std::unordered_map<Stmt *, StateMachine>>
      state_machines;

 public:
  using VariableOptimize::visit;

  StateMachine &get_state_machine(Stmt *stmt) override {
    return state_machines[stmt->parent][stmt];
  }

  void modify_all_state_machines(void (StateMachine::*func)()) override {
    for (auto &i : state_machines) {
      for (auto &j : i.second) {
        (j.second.*func)();
      }
    }
  }

  void clear() override {
    state_machines.clear();
  }

  void visit(AllocaStmt *stmt) override {
    state_machines[stmt->parent].insert(
        std::make_pair(stmt, StateMachine(stmt)));
  }

  void visit(AtomicOpStmt *stmt) override {
    if (!stmt->dest->is<AllocaStmt>())
      return;
    if (maybe_run)
      get_state_machine(stmt->dest).maybe_atomic_op();
    else
      get_state_machine(stmt->dest).atomic_op(stmt);
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
    auto origin = state_machines;
    modify_all_state_machines(&StateMachine::begin_if_or_loop);
    if (if_stmt->true_statements) {
      if_stmt->true_statements->accept(this);
    }
    auto true_branch = std::move(state_machines);

    state_machines = origin;
    modify_all_state_machines(&StateMachine::begin_if_or_loop);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
    auto false_branch = std::move(state_machines);

    state_machines = std::move(origin);
    for (auto &i : state_machines) {
      auto &true_branch_block = true_branch[i.first];
      auto &false_branch_block = false_branch[i.first];
      for (auto &j : i.second) {
        j.second.merge_from_if(true_branch_block[j.first],
                               false_branch_block[j.first]);
      }
    }
  }

  void visit_loop(Block *body, const std::vector<Stmt *> &loop_vars) override {
    if (maybe_run) {
      body->accept(this);
      return;
    }

    for (auto &loop_var : loop_vars) {
      get_state_machine(loop_var).mark_as_loop_var();
    }
    auto origin = state_machines;
    modify_all_state_machines(&StateMachine::begin_if_or_loop);
    maybe_run = true;
    body->accept(this);
    maybe_run = false;
    body->accept(this);
    for (auto &i : origin) {
      auto &loop_block = state_machines[i.first];
      for (auto &j : i.second) {
        j.second.merge_from_loop(loop_block[j.first]);
      }
    }
    state_machines = std::move(origin);
  }

  void visit(Block *block) override {
    state_machines.insert(
        std::make_pair(block, std::unordered_map<Stmt *, StateMachine>()));

    for (auto &stmt : block->statements) {
      stmt->accept(this);
    }
    if (!maybe_run) {
      for (auto &it : state_machines[block]) {
        it.second.finalize();
      }
    }
    state_machines.erase(block);
  }
};

class GlobalTempOptimize : public VariableOptimize {
 private:
  std::unordered_map<std::size_t, StateMachine> state_machines;
 public:
  using VariableOptimize::visit;

  StateMachine &get_state_machine(Stmt *stmt) override {
    return state_machines[stmt->as<GlobalTemporaryStmt>()->offset];
  }

  void modify_all_state_machines(void (StateMachine::*func)()) override {
    for (auto &i : state_machines) {
      (i.second.*func)();
    }
  }

  void clear() override {
    state_machines.clear();
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    if (state_machines.find(stmt->offset) == state_machines.end())
      state_machines.insert(std::make_pair(stmt->offset, StateMachine(stmt, false)));
  }

  void visit(AtomicOpStmt *stmt) override {
    if (!stmt->dest->is<GlobalTemporaryStmt>())
      return;
    if (maybe_run)
      get_state_machine(stmt->dest).maybe_atomic_op();
    else
      get_state_machine(stmt->dest).atomic_op(stmt);
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (!stmt->ptr->is<GlobalTemporaryStmt>())
      return;
    if (maybe_run)
      get_state_machine(stmt->ptr).maybe_store(stmt);
    else
      get_state_machine(stmt->ptr).store(stmt);
  }

  void visit(GlobalLoadStmt *stmt) override {
    if (!stmt->ptr->is<GlobalTemporaryStmt>())
      return;
    if (maybe_run)
      get_state_machine(stmt->ptr).maybe_load();
    else
      get_state_machine(stmt->ptr).load(stmt);
  }

  void visit(IfStmt *if_stmt) override {
    auto origin = state_machines;
    modify_all_state_machines(&StateMachine::begin_if_or_loop);
    if (if_stmt->true_statements) {
      if_stmt->true_statements->accept(this);
    }
    auto true_branch = std::move(state_machines);

    state_machines = origin;
    modify_all_state_machines(&StateMachine::begin_if_or_loop);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
    auto false_branch = std::move(state_machines);

    state_machines = std::move(origin);
    for (auto &it : state_machines) {
      it.second.merge_from_if(true_branch[it.first],
                              false_branch[it.first]);
    }
    for (auto &it : true_branch) {
      if (state_machines.find(it.first) == state_machines.end())
        state_machines.insert(it);
    }
    for (auto &it : false_branch) {
      if (state_machines.find(it.first) == state_machines.end())
        state_machines.insert(it);
    }
  }

  void visit_loop(Block *body, const std::vector<Stmt *> &loop_vars) override {
    if (maybe_run) {
      body->accept(this);
      return;
    }

    auto origin = state_machines;
    modify_all_state_machines(&StateMachine::begin_if_or_loop);
    maybe_run = true;
    body->accept(this);
    maybe_run = false;
    body->accept(this);
    for (auto &it : origin) {
      it.second.merge_from_loop(state_machines[it.first]);
    }
    for (auto &it : state_machines) {
      if (origin.find(it.first) == origin.end()) {
        StateMachine state_machine(it.second.get_var(), false);
        state_machine.merge_from_loop(it.second);
        origin.insert(std::make_pair(it.first, state_machine));
      }
    }
    state_machines = std::move(origin);
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->task_type == stmt->range_for) {
      TI_ASSERT(!maybe_run);
      if (!stmt->const_begin) {
        TI_ASSERT(state_machines.find(stmt->begin_offset) !=
                  state_machines.end());
        state_machines[stmt->begin_offset].load();
      }
      if (!stmt->const_end) {
        TI_ASSERT(state_machines.find(stmt->end_offset) !=
                  state_machines.end());
        state_machines[stmt->end_offset].load();
      }
    }
    if (stmt->body) {
      modify_all_state_machines(&StateMachine::begin_offload);
      stmt->body->accept(this);
    }
  }
};


namespace irpass {
void variable_optimization(IRNode *root) {
  AllocaOptimize alloca_optimizer;
  alloca_optimizer.run(root);
//  std::cout << "before\n";
//  print(root);
//  analysis::verify(root);
  GlobalTempOptimize global_temp_optimizer;
  global_temp_optimizer.run(root);
//  std::cout << "after\n";
//  print(root);
//  analysis::verify(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
