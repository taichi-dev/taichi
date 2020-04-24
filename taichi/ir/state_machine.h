#pragma once

#include "taichi/ir/ir.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

class StateMachine {
 private:
  Stmt *var;
  std::unordered_set<AtomicOpStmt *> *used_atomics;
  
 public:
  // If neither stored nor loaded (nor used as operands in masks/loop_vars),
  // we can safely delete the variable.
  bool stored;  // Is this variable ever stored (or atomic-operated)?
  bool loaded;  // Is this variable ever loaded (or atomic-operated)?

  Stmt *last_store;

  // last_store_valid: Can we do store-forwarding?
  // When the last store is conditional, last_store_invalid is false,
  // and last_store is set to the last store of one of the branches.
  bool last_store_valid;

  // last_store_loaded: Is the last store ever loaded? If not, eliminate it.
  bool last_store_loaded;

  AtomicOpStmt *last_atomic;

  // last_atomic_eliminable: Can we eliminate last_atomic if no statements
  // include it as an operand?
  bool last_atomic_eliminable;

  // Are we inside a loop which is inside the variable's scope?
  // outside_loop: No
  // inside_loop_may_have_stores: Yes
  // inside_loop_no_stores: Yes, but we've already checked that there are no
  //                        local stores in the loop and before the loop
  //                        (so that we can optimize local loads to const [0]).
  enum IsInsideLoop {
    outside_loop,
    inside_loop_may_have_stores,
    inside_loop_no_stores
  };
  IsInsideLoop is_inside_loop;

  // Are we in an if branch but haven't stored this variable?
  bool in_if_branch_but_not_stored;
  // Is this variable ever loaded before the first store in the current
  // if branch? This is for determining whether we can eliminate the last store
  // before the IfStmt.
  bool loaded_before_first_store_in_current_if_branch;

  explicit StateMachine(Stmt *var,
                        std::unordered_set<AtomicOpStmt *> *used_atomics);

  void store(Stmt *store_stmt);
  void atomic_op(AtomicOpStmt *stmt);
};

TLANG_NAMESPACE_END
