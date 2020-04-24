#pragma once

#include "taichi/ir/ir.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

class StateMachine {
 private:
  Stmt *var;
  static std::unique_ptr<std::unordered_set<AtomicOpStmt *>> used_atomics;

 public:
  // If neither stored nor loaded (nor used as operands in masks/loop_vars),
  // we can safely delete this variable if it's an alloca or a global temp.
  bool maybe_stored;  // Is this variable ever stored (or atomic-operated)?
  bool maybe_loaded;  // Is this variable ever loaded (or atomic-operated)?

  Stmt *last_store;

  // last_store_forwardable: Can we do store-forwarding?
  bool last_store_forwardable;

  // last_store_eliminable: Can we eliminate last_store?
  bool last_store_eliminable;

  AtomicOpStmt *last_atomic;

  // last_atomic_eliminable: Can we eliminate last_atomic?
  bool last_atomic_eliminable;

  // Are we in an if branch or a loop but haven't *definitely* stored this
  // variable? If yes, we can't eliminate the last store or AtomicOpStmt.
  bool in_if_or_loop_but_not_definitely_stored;
  // Is this variable ever loaded before the first *definite* store in the
  // current if branch? This is ONLY for determining whether we can eliminate
  // the last store before the IfStmt.
  bool maybe_loaded_before_first_definite_store_in_current_if_or_loop;

  explicit StateMachine(Stmt *var);

  // This must be called before using StateMachine to eliminate AtomicOpStmts.
  static void rebuild_atomics_usage(IRNode *root);

  void atomic_op(AtomicOpStmt *stmt);
  void store(Stmt *store_stmt);
  void load(Stmt *load_stmt);

  void maybe_atomic_op(AtomicOpStmt *);
  void maybe_store(Stmt *store_stmt);
  void maybe_load(Stmt *);
};

TLANG_NAMESPACE_END

/*
 * a = 1
 * for(...)
 *   b = a (x)
 *
 * a = 1 (x)
 * for(...)
 *   a = 2
 *
 * a = 1
 * for(...)
 *   b = a
 *   a = 2
 *
 * if(...)
 *  ...
 *  a = 1
 *  ...
 * else
 *  a = 1
 * b = a
 */
