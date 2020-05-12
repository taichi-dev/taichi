#include "state_machine.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/analysis.h"

TLANG_NAMESPACE_BEGIN

std::unique_ptr<std::unordered_set<AtomicOpStmt *>> StateMachine::used_atomics;

StateMachine::StateMachine(Stmt *var, bool zero_initialized)
    : var(var),
      stored(never),
      stored_in_this_if_or_loop(never),
      loaded(never),
      loaded_in_this_if_or_loop(never),
      last_store(nullptr),
      last_store_forwardable(false),
      last_store_eliminable(false),
      last_atomic(nullptr),
      last_atomic_eliminable(false),
      maybe_loaded_before_first_definite_store_in_this_if_or_loop(false) {
  if (!zero_initialized)
    stored = stored_in_this_if_or_loop = maybe;
}

bool StateMachine::same_data(Stmt *store_stmt1, Stmt *store_stmt2) {
  if (store_stmt1->is<LocalStoreStmt>()) {
    if (!store_stmt2->is<LocalStoreStmt>())
      return false;
    return irpass::analysis::same_statements(
        store_stmt1->as<LocalStoreStmt>()->data,
        store_stmt2->as<LocalStoreStmt>()->data);
  } else {
    if (!store_stmt2->is<GlobalStoreStmt>())
      return false;
    return irpass::analysis::same_statements(
        store_stmt1->as<GlobalStoreStmt>()->data,
        store_stmt2->as<GlobalStoreStmt>()->data);
  }
}

StateMachine::State StateMachine::merge_either_a_or_b(
    const StateMachine::State &a,
    const StateMachine::State &b) {
  if (a == definitely && b == definitely)
    return definitely;
  if (a != never || b != never)
    return maybe;
  return never;
}

StateMachine::State StateMachine::merge_a_and_b(const StateMachine::State &a,
                                                const StateMachine::State &b) {
  if (a == definitely || b == definitely)
    return definitely;
  if (a == maybe || b == maybe)
    return maybe;
  return never;
}

StateMachine::State StateMachine::merge_a_and_maybe_b(
    const StateMachine::State &a,
    const StateMachine::State &b) {
  if (a == definitely)
    return definitely;
  if (a == maybe || b != never)
    return maybe;
  return never;
}

void StateMachine::rebuild_atomics_usage(IRNode *root) {
  used_atomics = irpass::analysis::gather_used_atomics(root);
}

void StateMachine::atomic_op(AtomicOpStmt *stmt) {
  // This statement is loading the last store, so we can't eliminate it.
  if (stored_in_this_if_or_loop != definitely)
    maybe_loaded_before_first_definite_store_in_this_if_or_loop = true;

  stored = stored_in_this_if_or_loop = definitely;
  loaded = loaded_in_this_if_or_loop = definitely;

  last_store = nullptr;
  last_store_forwardable = false;
  last_store_eliminable = false;

  TI_ASSERT(used_atomics);
  last_atomic = stmt;
  last_atomic_eliminable = used_atomics->find(stmt) == used_atomics->end();
}

void StateMachine::store(Stmt *store_stmt) {
  TI_ASSERT(store_stmt->is<LocalStoreStmt>() ||
            store_stmt->is<GlobalStoreStmt>());
  if (last_store && last_store_eliminable &&
      stored_in_this_if_or_loop == definitely) {
    // The last store is never loaded.
    last_store->parent->erase(last_store);
    throw IRModified();
  }
  if (last_atomic && last_atomic_eliminable &&
      stored_in_this_if_or_loop == definitely) {
    // The last AtomicOpStmt is never used.
    last_atomic->parent->erase(last_atomic);
    throw IRModified();
  }
  if (last_store_forwardable && same_data(last_store, store_stmt)) {
    // This store is useless.
    store_stmt->parent->erase(store_stmt);
    throw IRModified();
  }
  stored = stored_in_this_if_or_loop = definitely;

  last_store = store_stmt;
  last_store_forwardable = true;
  last_store_eliminable = true;

  last_atomic = nullptr;
  last_atomic_eliminable = false;
}

void StateMachine::load(Stmt *load_stmt) {
  // The load_stmt == nullptr case is only for an offloaded range_for loading
  // global temps via begin_offset and end_offset.
  if (load_stmt)
    TI_ASSERT(load_stmt->is<LocalLoadStmt>() ||
              load_stmt->is<GlobalLoadStmt>());
  if (stored_in_this_if_or_loop != definitely)
    maybe_loaded_before_first_definite_store_in_this_if_or_loop = true;
  loaded = loaded_in_this_if_or_loop = definitely;
  last_store_eliminable = false;
  last_atomic_eliminable = false;
  if (!load_stmt)
    return;

  if (stored == never) {
    auto zero = load_stmt->insert_after_me(Stmt::make<ConstStmt>(
        LaneAttribute<TypedConstant>(load_stmt->ret_type.data_type)));
    zero->repeat(load_stmt->width());
    int current_stmt_id = load_stmt->parent->locate(load_stmt);
    load_stmt->replace_with(zero);
    load_stmt->parent->erase(current_stmt_id);
    throw IRModified();
  }
  if (last_store_forwardable) {
    // store-forwarding
    if (last_store->is<LocalStoreStmt>())
      load_stmt->replace_with(last_store->as<LocalStoreStmt>()->data);
    else
      load_stmt->replace_with(last_store->as<GlobalStoreStmt>()->data);
    load_stmt->parent->erase(load_stmt);
    throw IRModified();
  }
}

void StateMachine::continue_or_break() {
  last_store_eliminable = false;
  last_atomic_eliminable = false;
}

void StateMachine::maybe_atomic_op() {
  if (stored_in_this_if_or_loop != definitely)
    maybe_loaded_before_first_definite_store_in_this_if_or_loop = true;
  if (stored == never)
    stored = maybe;
  if (stored_in_this_if_or_loop == never)
    stored_in_this_if_or_loop = maybe;
  if (loaded == never)
    loaded = maybe;
  if (loaded_in_this_if_or_loop == never)
    loaded_in_this_if_or_loop = maybe;

  last_store = nullptr;
  last_store_forwardable = false;
  last_store_eliminable = false;

  last_atomic = nullptr;
  last_atomic_eliminable = false;
}

void StateMachine::maybe_store(Stmt *store_stmt) {
  TI_ASSERT(store_stmt->is<LocalStoreStmt>() ||
            store_stmt->is<GlobalStoreStmt>());
  if (stored == never)
    stored = maybe;
  if (stored_in_this_if_or_loop == never)
    stored_in_this_if_or_loop = maybe;

  if (last_store_forwardable) {
    last_store_forwardable = same_data(last_store, store_stmt);
  }
}

void StateMachine::maybe_load() {
  if (stored_in_this_if_or_loop != definitely)
    maybe_loaded_before_first_definite_store_in_this_if_or_loop = true;
  if (loaded == never)
    loaded = maybe;
  if (loaded_in_this_if_or_loop == never)
    loaded_in_this_if_or_loop = maybe;
  last_store_eliminable = false;
  last_atomic_eliminable = false;
}

void StateMachine::mark_as_loop_var() {
  stored = stored_in_this_if_or_loop = definitely;
  loaded = loaded_in_this_if_or_loop = definitely;
  last_store = nullptr;
  last_store_forwardable = false;
  last_store_eliminable = false;
  last_atomic = nullptr;
  last_atomic_eliminable = false;
  maybe_loaded_before_first_definite_store_in_this_if_or_loop = false;
}

void StateMachine::begin_offload() {
  last_store_forwardable = false;
}

void StateMachine::begin_if_or_loop() {
  stored_in_this_if_or_loop = never;
  loaded_in_this_if_or_loop = never;
  maybe_loaded_before_first_definite_store_in_this_if_or_loop = false;
}

void StateMachine::merge_from_if(const StateMachine &true_branch,
                                 const StateMachine &false_branch) {
  if (last_store && last_store_eliminable &&
      true_branch.stored_in_this_if_or_loop == definitely &&
      !true_branch
           .maybe_loaded_before_first_definite_store_in_this_if_or_loop &&
      false_branch.stored_in_this_if_or_loop == definitely &&
      !false_branch
           .maybe_loaded_before_first_definite_store_in_this_if_or_loop) {
    // The last store is never loaded.
    last_store->parent->erase(last_store);
    throw IRModified();
  }
  if (last_atomic && last_atomic_eliminable &&
      true_branch.stored_in_this_if_or_loop == definitely &&
      !true_branch
           .maybe_loaded_before_first_definite_store_in_this_if_or_loop &&
      false_branch.stored_in_this_if_or_loop == definitely &&
      !false_branch
           .maybe_loaded_before_first_definite_store_in_this_if_or_loop) {
    // The last AtomicOpStmt is never used.
    last_atomic->parent->erase(last_atomic);
    throw IRModified();
  }

  if (stored_in_this_if_or_loop != definitely) {
    maybe_loaded_before_first_definite_store_in_this_if_or_loop =
        maybe_loaded_before_first_definite_store_in_this_if_or_loop ||
        true_branch
            .maybe_loaded_before_first_definite_store_in_this_if_or_loop ||
        false_branch
            .maybe_loaded_before_first_definite_store_in_this_if_or_loop;
  }

  stored = merge_either_a_or_b(true_branch.stored, false_branch.stored);
  stored_in_this_if_or_loop = merge_a_and_b(
      stored_in_this_if_or_loop,
      merge_either_a_or_b(true_branch.stored_in_this_if_or_loop,
                          false_branch.stored_in_this_if_or_loop));
  loaded = merge_either_a_or_b(true_branch.loaded, false_branch.loaded);
  loaded_in_this_if_or_loop = merge_a_and_b(
      loaded_in_this_if_or_loop,
      merge_either_a_or_b(true_branch.loaded_in_this_if_or_loop,
                          false_branch.loaded_in_this_if_or_loop));

  if (true_branch.last_store_forwardable &&
      false_branch.last_store_forwardable &&
      same_data(true_branch.last_store, false_branch.last_store)) {
    last_store_forwardable = true;
    if (last_store == true_branch.last_store ||
        last_store == false_branch.last_store) {
      // The last store didn't change.
      last_store_eliminable =
          last_store_eliminable &&
          true_branch.last_store == false_branch.last_store &&
          true_branch.last_store_eliminable &&
          false_branch.last_store_eliminable;
    } else {
      TI_ASSERT(true_branch.last_store != false_branch.last_store);
      // if $b
      //   $c : store $a <- v1
      // else
      //   $d : store $a <- v1
      // Maybe move them outside in the future?
      if (true_branch.last_store_eliminable) {
        last_store = true_branch.last_store;
        last_store_eliminable = true;
      } else {
        last_store = false_branch.last_store;
        last_store_eliminable = false_branch.last_store_eliminable;
      }
    }
  } else {
    last_store_forwardable = false;
    // We only care if we can eliminate the last store here.
    if (true_branch.last_store == last_store &&
        false_branch.last_store == last_store) {
      // The last store didn't change.
      last_store_eliminable = last_store_eliminable &&
                              true_branch.last_store_eliminable &&
                              false_branch.last_store_eliminable;
    } else {
      // The last store changed.
      bool current_eliminable =
          last_store && last_store_eliminable &&
          !true_branch
               .maybe_loaded_before_first_definite_store_in_this_if_or_loop &&
          !false_branch
               .maybe_loaded_before_first_definite_store_in_this_if_or_loop;
      bool true_eliminable = true_branch.last_store != last_store &&
                             true_branch.last_store != nullptr &&
                             true_branch.last_store_eliminable;
      bool false_eliminable = false_branch.last_store != last_store &&
                              false_branch.last_store != nullptr &&
                              false_branch.last_store_eliminable;
      if (true_eliminable) {
        last_store = true_branch.last_store;
        last_store_eliminable = true;
      } else if (false_eliminable) {
        last_store = false_branch.last_store;
        last_store_eliminable = true;
      } else if (current_eliminable) {
        last_store_eliminable = true;
      } else {
        // Neither branch provides a eliminable local store.
        last_store = nullptr;
        last_store_eliminable = false;
      }
    }
  }

  // We only care if we can eliminate the last AtomicOpStmt here.
  if (true_branch.last_atomic == last_atomic &&
      false_branch.last_atomic == last_atomic) {
    // The last AtomicOpStmt didn't change.
    last_atomic_eliminable = last_atomic_eliminable &&
                             true_branch.last_atomic_eliminable &&
                             false_branch.last_atomic_eliminable;
  } else {
    // The last store changed.
    bool current_eliminable =
        last_atomic && last_atomic_eliminable &&
        !true_branch
             .maybe_loaded_before_first_definite_store_in_this_if_or_loop &&
        !false_branch
             .maybe_loaded_before_first_definite_store_in_this_if_or_loop;
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
    } else if (current_eliminable) {
      last_atomic_eliminable = true;
    } else {
      // Neither branch provides a eliminable local store.
      last_atomic = nullptr;
      last_atomic_eliminable = false;
    }
  }
}

void StateMachine::merge_from_loop(const StateMachine &loop) {
  if (stored_in_this_if_or_loop != definitely) {
    maybe_loaded_before_first_definite_store_in_this_if_or_loop =
        maybe_loaded_before_first_definite_store_in_this_if_or_loop ||
        loop.maybe_loaded_before_first_definite_store_in_this_if_or_loop;
  }

  stored = merge_a_and_maybe_b(stored, loop.stored);
  stored_in_this_if_or_loop = merge_a_and_maybe_b(
      stored_in_this_if_or_loop, loop.stored_in_this_if_or_loop);
  loaded = merge_a_and_maybe_b(loaded, loop.loaded);
  loaded_in_this_if_or_loop = merge_a_and_maybe_b(
      loaded_in_this_if_or_loop, loop.loaded_in_this_if_or_loop);

  // We must be cautious here because of possible Continues and WhileControls.
  if (loop.stored_in_this_if_or_loop != never) {
    // Not forwardable if stored in the loop.
    if (loop.loaded_in_this_if_or_loop != never) {
      // Not eliminable if loaded in the loop.
      last_store = nullptr;
      last_store_forwardable = false;
      last_store_eliminable = false;
      last_atomic = nullptr;
      last_atomic_eliminable = false;
    } else {
      last_store = loop.last_store;
      last_store_forwardable = false;
      last_store_eliminable = loop.last_atomic_eliminable;
      last_atomic = loop.last_atomic;
      last_atomic_eliminable = loop.last_atomic_eliminable;
    }
  } else {
    if (loop.loaded_in_this_if_or_loop != never) {
      // Not eliminable if loaded in the loop.
      last_store_eliminable = false;
      last_atomic_eliminable = false;
    }
  }
}

void StateMachine::finalize() {
  if (last_store && last_store_eliminable) {
    // The last store is never loaded.
    last_store->parent->erase(last_store);
    throw IRModified();
  }
  if (last_atomic && last_atomic_eliminable) {
    // The last AtomicOpStmt is never used.
    last_atomic->parent->erase(last_atomic);
    throw IRModified();
  }
  if (stored == never && loaded == never) {
    // Never stored and never loaded.
    // For future vectorization, if it's an alloca, we need to check that
    // this alloca is not used as masks (this can be done by checking operands)
    // before eliminating it.
    var->parent->erase(var);
    throw IRModified();
  }
}

Stmt *StateMachine::get_var() const {
  return var;
}

TLANG_NAMESPACE_END
