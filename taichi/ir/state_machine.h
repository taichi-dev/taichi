#pragma once

#include "taichi/ir/ir.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// State machine for AllocaStmt/GlobalTemporaryStmt/GlobalPtrStmt.
class StateMachine {
 private:
  Stmt *var;
  static std::unique_ptr<std::unordered_set<AtomicOpStmt *>> used_atomics;

  bool same_data(Stmt *store_stmt1, Stmt *store_stmt2);

 public:
  // If neither stored nor loaded (nor used as operands in masks/loop_vars),
  // we can safely delete this variable if it's an alloca or a global temp.
  enum State {
    never,
    maybe,
    definitely
  };
  State stored;  // Is this variable ever stored (or atomic-operated)?
  State stored_in_this_if_or_loop;
  State loaded;  // Is this variable ever loaded (or atomic-operated)?
  State loaded_in_this_if_or_loop;

  static State merge(const State &a, const State &b) {
    if (a == definitely || b == definitely)
      return definitely;
    if (a == maybe || b == maybe)
      return maybe;
    return never;
  }

  Stmt *last_store;

  // last_store_forwardable: Can we do store-forwarding?
  bool last_store_forwardable;

  // last_store_eliminable: Can we eliminate last_store?
  bool last_store_eliminable;

  AtomicOpStmt *last_atomic;

  // last_atomic_eliminable: Can we eliminate last_atomic?
  bool last_atomic_eliminable;

  // Is this variable ever loaded before the first *definite* store in the
  // current if branch? This is ONLY for determining whether we can eliminate
  // the last store before the IfStmt.
  bool maybe_loaded_before_first_definite_store_in_this_if_or_loop;

  explicit StateMachine(Stmt *var);

  // This must be called before using StateMachine to eliminate AtomicOpStmts.
  static void rebuild_atomics_usage(IRNode *root);

  void atomic_op(AtomicOpStmt *stmt);
  void store(Stmt *store_stmt);
  void load(Stmt *load_stmt);

  void maybe_atomic_op(AtomicOpStmt *);
  void maybe_store(Stmt *store_stmt);
  void maybe_load(Stmt *);

  StateMachine new_instance_for_if_or_loop() const;
  void merge_from_if(const StateMachine &true_branch,
      const StateMachine &false_branch);

  // This should be called after the "maybe" pass of the loop.
  void merge_from_loop(const StateMachine &loop);
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
