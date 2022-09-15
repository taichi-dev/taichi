#include "taichi/transforms/loop_invariant_detector.h"

TLANG_NAMESPACE_BEGIN

class CacheLoopInvariantGlobalVars : public LoopInvariantDetector {
 public:
  using LoopInvariantDetector::visit;

  enum class CacheStatus { ReadOnly = 0, ReadWrite = 1 };

  std::unordered_map<Stmt *, std::pair<AllocaStmt *, CacheStatus>>
      cached_allocas;

  DelayedIRModifier modifier;

  explicit CacheLoopInvariantGlobalVars(const CompileConfig &config)
      : LoopInvariantDetector(config) {
  }

  void add_writeback(AllocaStmt *alloca_stmt, Stmt *stmt, Stmt *parent_stmt) {
    auto final_value = std::make_unique<LocalLoadStmt>(alloca_stmt);
    auto global_store =
        std::make_unique<GlobalStoreStmt>(stmt, final_value.get());
    modifier.insert_after(parent_stmt, std::move(global_store));
    modifier.insert_after(parent_stmt, std::move(final_value));
  }

  AllocaStmt *cache_global_to_local(Stmt *stmt,
                                    Stmt *parent_stmt,
                                    CacheStatus status) {
    if (auto &[cached, cached_status] = cached_allocas[stmt]; cached) {
      if (cached_status == CacheStatus::ReadOnly &&
          status == CacheStatus::ReadWrite) {
        add_writeback(cached, stmt, parent_stmt);
        cached_status = CacheStatus::ReadWrite;
      }
      return cached;
    }

    auto alloca_unique =
        std::make_unique<AllocaStmt>(stmt->ret_type.ptr_removed());
    auto alloca_stmt = alloca_unique.get();
    cached_allocas[stmt] = {alloca_stmt, status};
    auto new_global_load = std::make_unique<GlobalLoadStmt>(stmt);
    auto local_store =
        std::make_unique<LocalStoreStmt>(alloca_stmt, new_global_load.get());
    modifier.insert_before(parent_stmt, std::move(new_global_load));
    modifier.insert_before(parent_stmt, std::move(alloca_unique));
    modifier.insert_before(parent_stmt, std::move(local_store));

    if (status == CacheStatus::ReadWrite) {
      add_writeback(alloca_stmt, stmt, parent_stmt);
    }
    return alloca_stmt;
  }

  void visit(GlobalLoadStmt *stmt) override {
    if (is_loop_invariant(stmt->src, stmt->parent)) {
      auto alloca_stmt = cache_global_to_local(
          stmt->src, stmt->parent->parent_stmt, CacheStatus::ReadOnly);
      auto local_load = std::make_unique<LocalLoadStmt>(alloca_stmt);
      stmt->replace_usages_with(local_load.get());
      modifier.insert_before(stmt, std::move(local_load));
      modifier.erase(stmt);
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (is_loop_invariant(stmt->dest, stmt->parent)) {
      auto alloca_stmt = cache_global_to_local(
          stmt->dest, stmt->parent->parent_stmt, CacheStatus::ReadWrite);
      auto local_store =
          std::make_unique<LocalStoreStmt>(alloca_stmt, stmt->val);
      stmt->replace_usages_with(local_store.get());
      modifier.insert_before(stmt, std::move(local_store));
      modifier.erase(stmt);
    }
  }

  static bool run(IRNode *node, const CompileConfig &config) {
    bool modified = false;

    while (true) {
      CacheLoopInvariantGlobalVars eliminator(config);
      node->accept(&eliminator);
      if (eliminator.modifier.modify_ir())
        modified = true;
      else
        break;
    };

    return modified;
  }
};

namespace irpass {
bool cache_loop_invariant_global_vars(IRNode *root,
                                      const CompileConfig &config) {
  TI_AUTO_PROF;
  return CacheLoopInvariantGlobalVars::run(root, config);
}
}  // namespace irpass

TLANG_NAMESPACE_END
