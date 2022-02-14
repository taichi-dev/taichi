#include "taichi/analysis/gather_uniquely_accessed_pointers.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
#include <algorithm>

TLANG_NAMESPACE_BEGIN

class LoopUniqueStmtSearcher : public BasicStmtVisitor {
 private:
  // Constant values that don't change in the loop.
  std::unordered_set<Stmt *> loop_invariant_;

  // If loop_unique_[stmt] is -1, the value of stmt is unique among the
  // top-level loop.
  // If loop_unique_[stmt] is x >= 0, the value of stmt is unique to
  // the x-th loop index.
  std::unordered_map<Stmt *, int> loop_unique_;

 public:
  // The number of loop indices of the top-level loop.
  // -1 means uninitialized.
  int num_different_loop_indices{-1};
  using BasicStmtVisitor::visit;

  LoopUniqueStmtSearcher() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(LoopIndexStmt *stmt) override {
    if (stmt->loop->is<OffloadedStmt>())
      loop_unique_[stmt] = stmt->index;
  }

  void visit(LoopUniqueStmt *stmt) override {
    loop_unique_[stmt] = -1;
  }

  void visit(ConstStmt *stmt) override {
    loop_invariant_.insert(stmt);
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    loop_invariant_.insert(stmt);
  }

  void visit(UnaryOpStmt *stmt) override {
    if (loop_invariant_.count(stmt->operand) > 0) {
      loop_invariant_.insert(stmt);
    }

    // op loop-unique -> loop-unique
    if (loop_unique_.count(stmt->operand) > 0 &&
        (stmt->op_type == UnaryOpType::neg)) {
      // TODO: Other injective unary operations
      loop_unique_[stmt] = loop_unique_[stmt->operand];
    }
  }

  void visit(DecorationStmt *stmt) override {
    if (stmt->decoration.size() == 2 &&
        stmt->decoration[0] ==
            uint32_t(DecorationStmt::Decoration::loop_unique)) {
      if (loop_unique_.find(stmt->operand) == loop_unique_.end()) {
        loop_unique_[stmt->operand] = stmt->decoration[1];
        num_different_loop_indices =
            std::max(loop_unique_[stmt->operand] + 1, num_different_loop_indices);
      }
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    if (loop_invariant_.count(stmt->lhs) > 0 &&
        loop_invariant_.count(stmt->rhs) > 0) {
      loop_invariant_.insert(stmt);
    }

    // loop-unique op loop-invariant -> loop-unique
    if ((loop_unique_.count(stmt->lhs) > 0 &&
         loop_invariant_.count(stmt->rhs) > 0) &&
        (stmt->op_type == BinaryOpType::add ||
         stmt->op_type == BinaryOpType::sub ||
         stmt->op_type == BinaryOpType::bit_xor)) {
      // TODO: Other operations
      loop_unique_[stmt] = loop_unique_[stmt->lhs];
    }

    // loop-invariant op loop-unique -> loop-unique
    if ((loop_invariant_.count(stmt->lhs) > 0 &&
         loop_unique_.count(stmt->rhs) > 0) &&
        (stmt->op_type == BinaryOpType::add ||
         stmt->op_type == BinaryOpType::sub ||
         stmt->op_type == BinaryOpType::bit_xor)) {
      loop_unique_[stmt] = loop_unique_[stmt->rhs];
    }
  }

  bool is_ptr_indices_loop_unique(GlobalPtrStmt *stmt) const {
    // Check if the address is loop-unique, i.e., stmt contains
    // either a loop-unique index or all top-level loop indices.
    TI_ASSERT(num_different_loop_indices != -1);
    std::vector<int> loop_indices;
    loop_indices.reserve(stmt->indices.size());
    for (auto &index : stmt->indices) {
      auto loop_unique_index = loop_unique_.find(index);
      if (loop_unique_index != loop_unique_.end()) {
        if (loop_unique_index->second == -1) {
          // LoopUniqueStmt
          return true;
        } else {
          // LoopIndexStmt
          loop_indices.push_back(loop_unique_index->second);
        }
      }
    }
    std::sort(loop_indices.begin(), loop_indices.end());
    auto current_num_different_loop_indices =
        std::unique(loop_indices.begin(), loop_indices.end()) -
        loop_indices.begin();
    // for i, j in x:
    //     a[j, i] is loop-unique
    //     b[i, i] is not loop-unique (because there's no j)
    return current_num_different_loop_indices == num_different_loop_indices;
  }

  bool is_ptr_indices_loop_unique(ExternalPtrStmt *stmt) const {
    // Check if the address is loop-unique, i.e., stmt contains
    // either a loop-unique index or all top-level loop indices.
    TI_ASSERT(num_different_loop_indices != -1);
    std::vector<int> loop_indices;
    loop_indices.reserve(stmt->indices.size());
    for (auto &index : stmt->indices) {
      auto loop_unique_index = loop_unique_.find(index);
      if (loop_unique_index != loop_unique_.end()) {
        if (loop_unique_index->second == -1) {
          // LoopUniqueStmt
          return true;
        } else {
          // LoopIndexStmt
          loop_indices.push_back(loop_unique_index->second);
        }
      }
    }
    std::sort(loop_indices.begin(), loop_indices.end());
    auto current_num_different_loop_indices =
        std::unique(loop_indices.begin(), loop_indices.end()) -
        loop_indices.begin();

    // for i, j in x:
    //     a[j, i] is loop-unique
    //     b[i, i] is not loop-unique (because there's no j)
    //     c[j, i, 1] is loop-unique
    return current_num_different_loop_indices == num_different_loop_indices;
  }
};

class UniquelyAccessedSNodeSearcher : public BasicStmtVisitor {
 private:
  LoopUniqueStmtSearcher loop_unique_stmt_searcher_;

  // Search SNodes that are uniquely accessed, i.e., accessed by
  // one GlobalPtrStmt (or by definitely-same-address GlobalPtrStmts),
  // and that GlobalPtrStmt's address is loop-unique.
  std::unordered_map<const SNode *, GlobalPtrStmt *> accessed_pointer_;
  std::unordered_map<const SNode *, GlobalPtrStmt *> rel_access_pointer_;

  // Search any_arrs that are uniquely accessed. Maps: ArgID -> ExternalPtrStmt
  std::unordered_set<ExternalPtrStmt *> accessed_arr_pointer_;

 public:
  using BasicStmtVisitor::visit;

  UniquelyAccessedSNodeSearcher() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(GlobalPtrStmt *stmt) override {
    // mesh-for loop unique
    if (stmt->indices.size() == 1 &&
        stmt->indices[0]->is<MeshIndexConversionStmt>()) {
      auto idx = stmt->indices[0]->as<MeshIndexConversionStmt>()->idx;
      while (idx->is<MeshIndexConversionStmt>()) {  // special case: l2g +
                                                    // g2r
        idx = idx->as<MeshIndexConversionStmt>()->idx;
      }
      if (idx->is<LoopIndexStmt>() &&
          idx->as<LoopIndexStmt>()->is_mesh_index()) {  // from-end access
        for (auto &snode : stmt->snodes.data) {
          if (rel_access_pointer_.find(snode) ==
              rel_access_pointer_.end()) {  // not accessed by neibhours yet
            accessed_pointer_[snode] = stmt;
          } else {  // accessed by neibhours, so it's not unique
            accessed_pointer_[snode] = nullptr;
          }
        }
      } else {  // to-end access
        for (auto &snode : stmt->snodes.data) {
          rel_access_pointer_[snode] = stmt;
          accessed_pointer_[snode] =
              nullptr;  // from-end access should not be unique
        }
      }
    }
    // Range-for / struct-for
    for (auto &snode : stmt->snodes.data) {
      auto accessed_ptr = accessed_pointer_.find(snode);
      if (accessed_ptr == accessed_pointer_.end()) {
        if (loop_unique_stmt_searcher_.is_ptr_indices_loop_unique(stmt)) {
          accessed_pointer_[snode] = stmt;
        } else {
          accessed_pointer_[snode] = nullptr;  // not loop-unique
        }
      } else {
        if (!irpass::analysis::definitely_same_address(accessed_ptr->second,
                                                       stmt)) {
          accessed_ptr->second = nullptr;  // not uniquely accessed
        }
      }
    }
  }

  void visit(ExternalPtrStmt *stmt) override {
    // A memory location of an ExternalPtrStmt depends on the indices
    // If the accessed indices are loop unique,
    // the accessed memory location is loop unique
    if (loop_unique_stmt_searcher_.is_ptr_indices_loop_unique(stmt)) {
      accessed_arr_pointer_.insert(stmt);
    }
  }

  static std::pair<std::unordered_map<const SNode *, GlobalPtrStmt *>,
                   std::unordered_set<ExternalPtrStmt *>>
  run(IRNode *root) {
    TI_ASSERT(root->is<OffloadedStmt>());
    auto offload = root->as<OffloadedStmt>();
    UniquelyAccessedSNodeSearcher searcher;
    if (offload->task_type == OffloadedTaskType::range_for ||
        offload->task_type == OffloadedTaskType::mesh_for) {
      searcher.loop_unique_stmt_searcher_.num_different_loop_indices = 1;
    } else if (offload->task_type == OffloadedTaskType::struct_for) {
      searcher.loop_unique_stmt_searcher_.num_different_loop_indices =
          offload->snode->num_active_indices;
    } else {
      // serial
      searcher.loop_unique_stmt_searcher_.num_different_loop_indices = 0;
    }
    root->accept(&searcher.loop_unique_stmt_searcher_);
    root->accept(&searcher);
    return std::make_pair(searcher.accessed_pointer_, searcher.accessed_arr_pointer_);
  }
};

class UniquelyAccessedBitStructGatherer : public BasicStmtVisitor {
 private:
  std::unordered_map<OffloadedStmt *,
                     std::unordered_map<const SNode *, GlobalPtrStmt *>>
      result_;

 public:
  using BasicStmtVisitor::visit;

  UniquelyAccessedBitStructGatherer() {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->task_type == OffloadedTaskType::range_for ||
        stmt->task_type == OffloadedTaskType::mesh_for ||
        stmt->task_type == OffloadedTaskType::struct_for) {
      auto &loop_unique_bit_struct = result_[stmt];
      auto loop_unique_ptr =
          irpass::analysis::gather_uniquely_accessed_pointers(stmt).first;
      for (auto &it : loop_unique_ptr) {
        auto *snode = it.first;
        auto *ptr1 = it.second;
        if (ptr1 != nullptr && ptr1->indices.size() > 0 &&
            ptr1->indices[0]->is<MeshIndexConversionStmt>()) {
          continue;
        }
        if (snode->is_bit_level) {
          // Find the nearest non-bit-level ancestor
          while (snode->is_bit_level) {
            snode = snode->parent;
          }
          // Check whether uniquely accessed
          auto accessed_ptr = loop_unique_bit_struct.find(snode);
          if (accessed_ptr == loop_unique_bit_struct.end()) {
            loop_unique_bit_struct[snode] = ptr1;
          } else {
            if (ptr1 == nullptr) {
              accessed_ptr->second = nullptr;
              continue;
            }
            auto *ptr2 = accessed_ptr->second;
            TI_ASSERT(ptr1->indices.size() == ptr2->indices.size());
            for (int id = 0; id < (int)ptr1->indices.size(); id++) {
              if (!irpass::analysis::same_value(ptr1->indices[id],
                                                ptr2->indices[id])) {
                accessed_ptr->second = nullptr;  // not uniquely accessed
              }
            }
          }
        }
      }
    }
    // Do not dive into OffloadedStmt
  }

  static std::unordered_map<OffloadedStmt *,
                            std::unordered_map<const SNode *, GlobalPtrStmt *>>
  run(IRNode *root) {
    UniquelyAccessedBitStructGatherer gatherer;
    root->accept(&gatherer);
    return gatherer.result_;
  }
};

const std::string GatherUniquelyAccessedBitStructsPass::id =
    "GatherUniquelyAccessedBitStructsPass";

namespace irpass::analysis {
std::pair<std::unordered_map<const SNode *, GlobalPtrStmt *>,
          std::unordered_set<ExternalPtrStmt *>>
gather_uniquely_accessed_pointers(IRNode *root) {
  // TODO: What about SNodeOpStmts?
  return UniquelyAccessedSNodeSearcher::run(root);
}

void gather_uniquely_accessed_bit_structs(IRNode *root, AnalysisManager *amgr) {
  amgr->put_pass_result<GatherUniquelyAccessedBitStructsPass>(
      {UniquelyAccessedBitStructGatherer::run(root)});
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
