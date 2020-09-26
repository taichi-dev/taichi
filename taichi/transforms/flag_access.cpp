#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

// Flag accesses to be either weak (non-activating) or strong (activating)
class FlagAccess : public IRVisitor {
 public:
  FlagAccess(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
    node->accept(this);
  }

  void visit(Block *stmt_list) {  // block itself has no id
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(IfStmt *if_stmt) {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) {
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) {
    for_stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) {
    stmt->all_blocks_accept(this);
  }

  // Assuming pointers will be visited before global load/st
  void visit(GlobalPtrStmt *stmt) {
    stmt->activate = false;
  }

  void visit(GlobalStoreStmt *stmt) {
    if (stmt->ptr->is<GlobalPtrStmt>()) {
      stmt->ptr->as<GlobalPtrStmt>()->activate = true;
    }
  }

  void visit(AtomicOpStmt *stmt) {
    if (stmt->dest->is<GlobalPtrStmt>()) {
      stmt->dest->as<GlobalPtrStmt>()->activate = true;
    }
  }
};

// For struct fors, weaken accesses on variables currently being looped over
// E.g.
// for i in x:
//   x[i] = 0
// Here although we are writing to x[i], but i will only loop over active
// elements of x. So we don't need one more activation. Note the indices of x
// accesses must be loop indices for this optimization to be correct.

class WeakenAccess : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  WeakenAccess(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
    current_struct_for = nullptr;
    current_offload = nullptr;
    node->accept(this);
  }

  void visit(Block *stmt_list) {  // block itself has no id
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(StructForStmt *stmt) {
    current_struct_for = stmt;
    stmt->body->accept(this);
    current_struct_for = nullptr;
  }

  void visit(OffloadedStmt *stmt) {
    current_offload = stmt;
    if (stmt->body)
      stmt->body->accept(this);
    current_offload = nullptr;
  }

  static SNode *least_sparse_ancestor(SNode *a) {
    while (a->type == SNodeType::place || a->type == SNodeType::dense) {
      a = a->parent;
    }
    return a;
  }

  static bool share_sparsity(SNode *a, SNode *b) {
    return least_sparse_ancestor(a) == least_sparse_ancestor(b);
  }

  void visit(GlobalPtrStmt *stmt) {
    if (stmt->activate) {
      bool is_struct_for =
          (current_offload &&
           current_offload->task_type == OffloadedStmt::TaskType::struct_for) ||
          current_struct_for;
      if (is_struct_for) {
        bool same_as_loop_snode = true;
        for (auto snode : stmt->snodes.data) {
          SNode *loop_snode = nullptr;
          if (current_struct_for) {
            loop_snode = current_struct_for->snode;
          } else {
            loop_snode = current_offload->snode;
          }
          TI_ASSERT(loop_snode);
          if (!share_sparsity(snode, loop_snode)) {
            same_as_loop_snode = false;
          }
          if (stmt->indices.size() == loop_snode->num_active_indices)
            for (int i = 0; i < loop_snode->num_active_indices; i++) {
              auto ind = stmt->indices[i];
              // TODO: vectorized cases?
              if (auto loop_var = ind->cast<LoopIndexStmt>()) {
                if (loop_var->index != i) {
                  same_as_loop_snode = false;
                }
              } else {
                same_as_loop_snode = false;
              }
            }
        }
        if (same_as_loop_snode)
          stmt->activate = false;
      }
    }
  }

 private:
  OffloadedStmt *current_offload;
  StructForStmt *current_struct_for;
};

namespace irpass {

void flag_access(IRNode *root) {
  TI_AUTO_PROF;
  FlagAccess flag_access(root);
  WeakenAccess weaken_access(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
