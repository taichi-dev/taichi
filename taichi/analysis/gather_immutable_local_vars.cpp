#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

namespace taichi::lang {

// The GatherImmutableLocalVars pass gathers all immutable local vars as input
// to the EliminateImmutableLocalVars pass. An immutable local var is an alloca
// which is stored only once (in the same block) and only loaded after that
// store.
class GatherImmutableLocalVars : public BasicStmtVisitor {
 private:
  using BasicStmtVisitor::visit;

  enum class AllocaStatus { kCreated = 0, kStoredOnce = 1, kInvalid = 2 };
  std::unordered_map<Stmt *, AllocaStatus> alloca_status_;

 public:
  explicit GatherImmutableLocalVars() {
    invoke_default_visitor = true;
  }

  void visit(AllocaStmt *stmt) override {
    TI_ASSERT(alloca_status_.find(stmt) == alloca_status_.end());
    alloca_status_[stmt] = AllocaStatus::kCreated;
  }

  void visit(LocalLoadStmt *stmt) override {
    if (stmt->src->is<AllocaStmt>()) {
      auto status_iter = alloca_status_.find(stmt->src);
      TI_ASSERT(status_iter != alloca_status_.end());
      if (status_iter->second == AllocaStatus::kCreated) {
        status_iter->second = AllocaStatus::kInvalid;
      }
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    if (stmt->dest->is<AllocaStmt>()) {
      auto status_iter = alloca_status_.find(stmt->dest);
      TI_ASSERT(status_iter != alloca_status_.end());
      if (stmt->parent != stmt->dest->parent ||
          status_iter->second == AllocaStatus::kStoredOnce ||
          stmt->val->ret_type.ptr_removed() !=
              stmt->dest->ret_type.ptr_removed()) {
        // FIXME: ptr_removed() is a workaround for the fact that
        //   the type of the alloca is the type of the element instead of the
        //   type of the pointer.
        status_iter->second = AllocaStatus::kInvalid;
      } else if (status_iter->second == AllocaStatus::kCreated) {
        status_iter->second = AllocaStatus::kStoredOnce;
      }
    }
  }

  void default_visit(Stmt *stmt) {
    for (auto &op : stmt->get_operands()) {
      if (op != nullptr && op->is<AllocaStmt>()) {
        auto status_iter = alloca_status_.find(op);
        TI_ASSERT(status_iter != alloca_status_.end());
        status_iter->second = AllocaStatus::kInvalid;
      }
    }
  }

  void visit(Stmt *stmt) override {
    default_visit(stmt);
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    default_visit(stmt);
  }

  static std::unordered_set<Stmt *> run(IRNode *node) {
    GatherImmutableLocalVars pass;
    node->accept(&pass);
    std::unordered_set<Stmt *> result;
    for (auto &[k, v] : pass.alloca_status_) {
      if (v == AllocaStatus::kStoredOnce) {
        result.insert(k);
      }
    }
    return result;
  }
};

namespace irpass::analysis {

std::unordered_set<Stmt *> gather_immutable_local_vars(IRNode *root) {
  return GatherImmutableLocalVars::run(root);
}

}  // namespace irpass::analysis

}  // namespace taichi::lang
