#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include <deque>
#include <set>

TLANG_NAMESPACE_BEGIN

class DemoteAtomics : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  OffloadedStmt *current_offloaded;

  DemoteAtomics() : BasicStmtVisitor() {
    current_offloaded = nullptr;
  }

  void visit(AtomicOpStmt *stmt) override {
    bool demote = false;
    bool is_local = false;
    if (current_offloaded && arch_is_cpu(current_offloaded->device) &&
        current_offloaded->num_cpu_threads == 1) {
      demote = true;
    }
    if (stmt->dest->is<AllocaStmt>()) {
      demote = true;
      is_local = true;
    }
    if (demote) {
      // replace atomics with load, add, store
      if (stmt->op_type == AtomicOpType::add) {
        auto ptr = stmt->dest;
        auto val = stmt->val;

        auto new_stmts = VecStatement();
        Stmt *load;
        if (is_local) {
          TI_ASSERT(stmt->width() == 1);
          load = new_stmts.push_back<LocalLoadStmt>(LocalAddress(ptr, 0));
          auto add =
              new_stmts.push_back<BinaryOpStmt>(BinaryOpType::add, load, val);
          new_stmts.push_back<LocalStoreStmt>(ptr, add);
        } else {
          load = new_stmts.push_back<GlobalLoadStmt>(ptr);
          auto add =
              new_stmts.push_back<BinaryOpStmt>(BinaryOpType::add, load, val);
          new_stmts.push_back<GlobalStoreStmt>(ptr, add);
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
        stmt->parent->replace_with(stmt, std::move(new_stmts),
                                   /*replace_usages=*/false);
        throw IRModified();
      }
    }
  }

  void visit(OffloadedStmt *stmt) override {
    current_offloaded = stmt;
    if (stmt->body) {
      stmt->body->accept(this);
    }
    current_offloaded = nullptr;
  }

  static void run(IRNode *node) {
    DemoteAtomics demoter;
    while (true) {
      try {
        node->accept(&demoter);
      } catch (IRModified) {
        continue;
      }
      break;
    }
  }
};

namespace irpass {

void demote_atomics(IRNode *root) {
  DemoteAtomics::run(root);
  typecheck(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
