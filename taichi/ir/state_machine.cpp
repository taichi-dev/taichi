#include "state_machine.h"


TLANG_NAMESPACE_BEGIN

StateMachine::StateMachine(Stmt *var)
    : var(var),
      maybe_stored(false),
      maybe_loaded(false),
      last_store(nullptr),
      last_store_forwardable(false),
      last_store_eliminable(false),
      last_atomic(nullptr),
      last_atomic_eliminable(false),
      in_if_or_loop_but_not_definitely_stored(false),
      maybe_loaded_before_first_definite_store_in_current_if_or_loop(false) {
  TI_ASSERT(var->is<AllocaStmt>() || var->is<GlobalTemporaryStmt>() ||
      var->is<GlobalPtrStmt>());
}

void StateMachine::rebuild_atomics_usage(IRNode *root) {
  used_atomics = irpass::analysis::gather_used_atomics(root);
}

void StateMachine::atomic_op(AtomicOpStmt *stmt) {
  // This statement is loading the last store, so we can't eliminate it.
  maybe_stored = true;
  maybe_loaded = true;

  last_store = nullptr;
  last_store_forwardable = false;
  last_store_eliminable = false;

  TI_ASSERT(used_atomics);
  last_atomic = stmt;
  last_atomic_eliminable = used_atomics->find(stmt) == used_atomics->end();

  if (in_if_or_loop_but_not_definitely_stored)
    maybe_loaded_before_first_definite_store_in_current_if_or_loop = true;
  in_if_or_loop_but_not_definitely_stored = false;
}

void StateMachine::store(Stmt *store_stmt) {
  TI_ASSERT(store_stmt->is<LocalStoreStmt>() ||
      store_stmt->is<GlobalStoreStmt>());
  if (last_store && last_store_eliminable && !in_if_or_loop_but_not_definitely_stored) {
    // The last store is never loaded.
    last_store->parent->erase(last_store);
    throw IRModified();
  }
  if (last_atomic && last_atomic_eliminable && !in_if_or_loop_but_not_definitely_stored) {
    // The last AtomicOpStmt is never used.
    last_atomic->parent->erase(last_atomic);
    throw IRModified();
  }
  maybe_stored = true;

  last_store = store_stmt;
  last_store_forwardable = true;
  last_store_eliminable = true;

  last_atomic = nullptr;
  last_atomic_eliminable = false;

  in_if_or_loop_but_not_definitely_stored = false;
}

void StateMachine::load(Stmt *load_stmt) {
  TI_ASSERT(load_stmt->is<LocalLoadStmt>() || load_stmt->is<GlobalLoadStmt>());
  maybe_loaded = true;
  last_store_eliminable = false;
  last_atomic_eliminable = false;
  if (in_if_or_loop_but_not_definitely_stored)
    maybe_loaded_before_first_definite_store_in_current_if_or_loop = true;

  if (!maybe_stored) {
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

void StateMachine::maybe_atomic_op(AtomicOpStmt *) {
  maybe_stored = true;
  maybe_loaded = true;

  last_store = nullptr;
  last_store_forwardable = false;
  last_store_eliminable = false;

  last_atomic = nullptr;
  last_atomic_eliminable = false;

  if (in_if_or_loop_but_not_definitely_stored)
    maybe_loaded_before_first_definite_store_in_current_if_or_loop = true;
}

void StateMachine::maybe_store(Stmt *store_stmt) {
  TI_ASSERT(store_stmt->is<LocalStoreStmt>() ||
      store_stmt->is<GlobalStoreStmt>());
  maybe_stored = true;

  if (last_store_forwardable) {
    if (last_store->is<LocalStoreStmt>()) {
      TI_ASSERT(store_stmt->is<LocalStoreStmt>());
      last_store_forwardable = irpass::analysis::same_statements(
          last_store->as<LocalStoreStmt>()->data,
          store_stmt->as<LocalStoreStmt>()->data);
    } else {
      TI_ASSERT(store_stmt->is<GlobalStoreStmt>());
      last_store_forwardable = irpass::analysis::same_statements(
          last_store->as<GlobalStoreStmt>()->data,
          store_stmt->as<GlobalStoreStmt>()->data);
    }
  }
}

void StateMachine::maybe_load(Stmt *) {
  maybe_loaded = true;
  last_store_eliminable = false;
  last_atomic_eliminable = false;
  if (in_if_or_loop_but_not_definitely_stored)
    maybe_loaded_before_first_definite_store_in_current_if_or_loop = true;
}

TLANG_NAMESPACE_END
