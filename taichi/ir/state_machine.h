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
  enum State { never, maybe, definitely };
  State stored;  // Is this variable ever stored (or atomic-operated)?
  State stored_in_this_if_or_loop;
  State loaded;  // Is this variable ever loaded (or atomic-operated)?
  State loaded_in_this_if_or_loop;

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

  StateMachine() {
    TI_ERROR("StateMachine constructor invoked with no parameters.")
  }
  explicit StateMachine(Stmt *var, bool zero_initialized);

  // This must be called before using StateMachine to eliminate AtomicOpStmts.
  static void rebuild_atomics_usage(IRNode *root);

  static State merge_either_a_or_b(const State &a, const State &b);
  static State merge_a_and_b(const State &a, const State &b);
  static State merge_a_and_maybe_b(const State &a, const State &b);

  void atomic_op(AtomicOpStmt *stmt);
  void store(Stmt *store_stmt);
  void load(Stmt *load_stmt = nullptr);

  void continue_or_break();

  void maybe_atomic_op();
  void maybe_store(Stmt *store_stmt);
  void maybe_load();

  void mark_as_loop_var();

  void begin_offload();
  void begin_if_or_loop();
  void merge_from_if(const StateMachine &true_branch,
                     const StateMachine &false_branch);
  void merge_from_loop(const StateMachine &loop);

  void finalize();

  Stmt *get_var() const;
};

TLANG_NAMESPACE_END
