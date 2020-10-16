#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

class LoopUniqueStmtSearcher : public BasicStmtVisitor {
 private:
  std::unordered_set<Stmt *> loop_invariant_;
  std::unordered_set<Stmt *> loop_unique_;

 public:
  using BasicStmtVisitor::visit;

  LoopUniqueStmtSearcher() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(LoopIndexStmt *stmt) override {
    if (stmt->loop->is<OffloadedStmt>())
      loop_unique_.insert(stmt);
  }

  void visit(LoopUniqueStmt *stmt) override {
    loop_unique_.insert(stmt);
  }

  void visit(ConstStmt *stmt) override {
    loop_invariant_.insert(stmt);
  }

  void visit(UnaryOpStmt *stmt) override {
    if (loop_invariant_.count(stmt->operand) > 0) {
      loop_invariant_.insert(stmt);
    }
    if (loop_unique_.count(stmt->operand) > 0 &&
        (stmt->op_type == UnaryOpType::neg)) {
      // TODO: Other injective unary operations
      loop_unique_.insert(stmt);
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    if (loop_invariant_.count(stmt->lhs) > 0 &&
        loop_invariant_.count(stmt->rhs) > 0) {
      loop_invariant_.insert(stmt);
    }
    if (((loop_unique_.count(stmt->lhs) > 0 &&
          loop_invariant_.count(stmt->rhs) > 0) ||
         (loop_invariant_.count(stmt->lhs) > 0 &&
          loop_unique_.count(stmt->rhs) > 0)) &&
        (stmt->op_type == BinaryOpType::add ||
         stmt->op_type == BinaryOpType::sub ||
         stmt->op_type == BinaryOpType::bit_xor)) {
      loop_unique_.insert(stmt);
    }
  }

  bool is_loop_unique(Stmt *stmt) const {
    return loop_unique_.count(stmt) > 0;
  }
};

class UniquelyAccessedSNodeSearcher : public BasicStmtVisitor {
 private:
  LoopUniqueStmtSearcher loop_unique_stmt_searcher_;
  std::unordered_map<SNode *, GlobalPtrStmt *> accessed_pointer_;

 public:
  using BasicStmtVisitor::visit;

  UniquelyAccessedSNodeSearcher() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(GlobalPtrStmt *stmt) override {
    for (auto &snode : stmt->snodes.data) {
      auto accessed_ptr = accessed_pointer_.find(snode);
      if (accessed_ptr == accessed_pointer_.end()) {
        accessed_pointer_[snode] = stmt;
        for (auto &index : stmt->indices) {
          if (!loop_unique_stmt_searcher_.is_loop_unique(index)) {
            accessed_pointer_[snode] = nullptr;  // not loop-unique
            break;
          }
        }
      } else {
        if (!irpass::analysis::definitely_same_address(accessed_ptr->second,
                                                       stmt)) {
          accessed_ptr->second = nullptr;  // not uniquely accessed
        }
      }
    }
  }

  static std::unordered_map<SNode *, GlobalPtrStmt *> run(IRNode *root) {
    TI_ASSERT(root->is<OffloadedStmt>());
    UniquelyAccessedSNodeSearcher searcher;
    root->accept(&searcher.loop_unique_stmt_searcher_);
    root->accept(&searcher);
    return std::move(searcher.accessed_pointer_);
  }
};

namespace irpass::analysis {
std::unordered_map<SNode *, GlobalPtrStmt *> gather_uniquely_accessed_pointers(
    IRNode *root) {
  return UniquelyAccessedSNodeSearcher::run(root);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
