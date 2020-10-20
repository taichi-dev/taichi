#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include <deque>
#include <set>

TLANG_NAMESPACE_BEGIN

class DemoteAtomics : public BasicStmtVisitor {
 private:
  std::unordered_map<SNode *, GlobalPtrStmt *> loop_unique_ptr_;

 public:
  using BasicStmtVisitor::visit;

  OffloadedStmt *current_offloaded;
  DelayedIRModifier modifier;

  DemoteAtomics() : BasicStmtVisitor() {
    current_offloaded = nullptr;
  }

  void visit(AtomicOpStmt *stmt) override {
    bool demote = false;
    bool is_local = false;
    if (current_offloaded) {
      if (arch_is_cpu(current_offloaded->device) &&
          current_offloaded->num_cpu_threads == 1) {
        demote = true;
      }
      if (stmt->dest->is<ThreadLocalPtrStmt>()) {
        demote = true;
      }
      if (current_offloaded->task_type == OffloadedTaskType::serial) {
        demote = true;
      }
      if (!demote &&
          (current_offloaded->task_type == OffloadedTaskType::range_for ||
           current_offloaded->task_type == OffloadedTaskType::struct_for) &&
          stmt->dest->is<GlobalPtrStmt>()) {
        demote = true;
        auto dest = stmt->dest->as<GlobalPtrStmt>();
        for (auto snode : dest->snodes.data) {
          if (loop_unique_ptr_[snode] == nullptr ||
              loop_unique_ptr_[snode]->indices.empty()) {
            // not uniquely accessed
            demote = false;
            break;
          }
        }
      }
    }
    if (stmt->dest->is<AllocaStmt>()) {
      demote = true;
      is_local = true;
    }
    if (demote) {
      // replace atomics with load, add, store
      auto bin_type = atomic_to_binary_op_type(stmt->op_type);
      auto ptr = stmt->dest;
      auto val = stmt->val;

      auto new_stmts = VecStatement();
      Stmt *load;
      if (is_local) {
        TI_ASSERT(stmt->width() == 1);
        load = new_stmts.push_back<LocalLoadStmt>(LocalAddress(ptr, 0));
        auto bin = new_stmts.push_back<BinaryOpStmt>(bin_type, load, val);
        new_stmts.push_back<LocalStoreStmt>(ptr, bin);
      } else {
        load = new_stmts.push_back<GlobalLoadStmt>(ptr);
        auto bin = new_stmts.push_back<BinaryOpStmt>(bin_type, load, val);
        new_stmts.push_back<GlobalStoreStmt>(ptr, bin);
      }
      // For a taichi program like `c = ti.atomic_add(a, b)`, the IR looks
      // like the following
      //
      // $c  = # lhs memory
      // $d  = atomic add($a, $b)
      // $e  : store [$c <- $d]
      //
      // If this gets demoted, the IR is translated into:
      //
      // $c  = # lhs memory
      // $d' = load $a             <-- added by demote_atomic
      // $e' = add $d' $b
      // $f  : store [$a <- $e']   <-- added by demote_atomic
      // $g  : store [$c <- ???]   <-- store the old value into lhs $c
      //
      // Naively relying on Block::replace_with() would incorrectly fill $f
      // into ???, because $f is a store stmt that doesn't have a return
      // value. The correct thing is to replace |stmt| $d with the loaded
      // old value $d'.
      // See also: https://github.com/taichi-dev/taichi/issues/332
      stmt->replace_with(load);
      modifier.replace_with(stmt, std::move(new_stmts),
                            /*replace_usages=*/false);
    }
  }

  void visit(OffloadedStmt *stmt) override {
    current_offloaded = stmt;
    if (stmt->task_type == OffloadedTaskType::range_for ||
        stmt->task_type == OffloadedTaskType::struct_for) {
      loop_unique_ptr_ =
          irpass::analysis::gather_uniquely_accessed_pointers(stmt);
    }
    // We don't need to visit TLS/BLS prologues/epilogues.
    if (stmt->body) {
      stmt->body->accept(this);
    }
    current_offloaded = nullptr;
  }

  static bool run(IRNode *node) {
    DemoteAtomics demoter;
    bool modified = false;
    while (true) {
      node->accept(&demoter);
      if (demoter.modifier.modify_ir()) {
        modified = true;
      } else {
        break;
      }
    }
    return modified;
  }
};

namespace irpass {

bool demote_atomics(IRNode *root) {
  TI_AUTO_PROF;
  bool modified = DemoteAtomics::run(root);
  type_check(root);
  return modified;
}

}  // namespace irpass

TLANG_NAMESPACE_END
