#include "state_machine.h"


TLANG_NAMESPACE_BEGIN

StateMachine::StateMachine(Stmt *var,
                           std::unordered_set<AtomicOpStmt *> *used_atomics)
    : var(var),
      used_atomics(used_atomics),
      stored(false),
      loaded(false),
      last_store(nullptr),
      last_store_valid(false),
      last_store_loaded(false),
      last_atomic(nullptr),
      last_atomic_eliminable(false),
      is_inside_loop(outside_loop),
      in_if_branch_but_not_stored(false),
      loaded_before_first_store_in_current_if_branch(false) {
  TI_ASSERT(var->is<AllocaStmt>() || var->is<GlobalTemporaryStmt>() ||
      var->is<GlobalPtrStmt>());
}

void StateMachine::store(Stmt *store_stmt) {
  TI_ASSERT(store_stmt->is<LocalStoreStmt>() ||
      store_stmt->is<GlobalStoreStmt>());
  if (last_store && !last_store_loaded && !in_if_branch_but_not_stored) {
    // The last store is never loaded.
    last_store->parent->erase(last_store);
    throw IRModified();
  }
  if (last_atomic && last_atomic_eliminable && !in_if_branch_but_not_stored) {
    // The last AtomicOpStmt is never used.
    last_atomic->parent->erase(last_atomic);
    throw IRModified();
  }
  stored = true;

  last_store = store_stmt;
  last_store_valid = true;
  last_store_loaded = false;

  last_atomic = nullptr;
  last_atomic_eliminable = false;

  in_if_branch_but_not_stored = false;
}

void StateMachine::atomic_op(AtomicOpStmt *stmt) {
  // This statement is loading the last store, so we can't eliminate it.
  stored = true;
  loaded = true;

  last_store = nullptr;
  last_store_valid = false;
  last_store_loaded = false;

  last_atomic = stmt;
  last_atomic_eliminable = used_atomics->find(stmt) == used_atomics->end();

  if (in_if_branch_but_not_stored)
    loaded_before_first_store_in_current_if_branch = true;
  in_if_branch_but_not_stored = false;
}

TLANG_NAMESPACE_END
