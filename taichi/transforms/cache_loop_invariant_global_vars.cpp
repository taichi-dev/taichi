#include "taichi/transforms/loop_invariant_detector.h"

TLANG_NAMESPACE_BEGIN

class CacheLoopInvariantGlobalVars : public LoopInvariantDetector {
 public:
  using LoopInvariantDetector::visit;

  enum class CacheStatus { Read = 1, Write = 2, ReadWrite = 3 };

  typedef std::unordered_map<Stmt *, std::pair<AllocaStmt *, CacheStatus>> CacheMap;
  std::stack<CacheMap> cached_maps;

  DelayedIRModifier modifier;

  explicit CacheLoopInvariantGlobalVars(const CompileConfig &config)
      : LoopInvariantDetector(config) {
  }

  void visit_loop(Block *body) override {
    cached_maps.emplace();
    LoopInvariantDetector::visit_loop(body);
    cached_maps.pop();
  }

  void add_writeback(AllocaStmt *alloca_stmt, Stmt *global_var) {
    auto final_value = std::make_unique<LocalLoadStmt>(alloca_stmt);
    auto global_store =
        std::make_unique<GlobalStoreStmt>(global_var, final_value.get());
    modifier.insert_after(current_loop_stmt(), std::move(global_store));
    modifier.insert_after(current_loop_stmt(), std::move(final_value));
  }

  void set_init_value(AllocaStmt *alloca_stmt, Stmt *global_var) {
    auto new_global_load = std::make_unique<GlobalLoadStmt>(global_var);
    auto local_store =
        std::make_unique<LocalStoreStmt>(alloca_stmt, new_global_load.get());
    modifier.insert_before(current_loop_stmt(), std::move(new_global_load));
    modifier.insert_before(current_loop_stmt(), std::move(local_store));
  }

  /*
   *
   */
  AllocaStmt *cache_global_to_local(Stmt *stmt,
                                    CacheStatus status) {
    if (auto &[cached, cached_status] = cached_maps.top()[stmt]; cached) {
      // The global variable has already been cached.
      if (cached_status == CacheStatus::Read &&
          status == CacheStatus::Write) {
        // If the
        add_writeback(cached, stmt);
        cached_status = CacheStatus::ReadWrite;
      }
      return cached;
    }

    auto alloca_unique =
        std::make_unique<AllocaStmt>(stmt->ret_type.ptr_removed());
    auto alloca_stmt = alloca_unique.get();
    modifier.insert_before(loop_blocks.top()->parent_stmt, std::move(alloca_unique));
    cached_maps.top()[stmt] = {alloca_stmt, status};

    if (status == CacheStatus::Read) {
      set_init_value(alloca_stmt, stmt);
    }

    if (status == CacheStatus::Write) {
      add_writeback(alloca_stmt, stmt);
    }
    return alloca_stmt;
  }

  void visit(GlobalLoadStmt *stmt) override {
    if (is_operand_loop_invariant_impl(stmt->src, stmt->parent)) {
      auto alloca_stmt = cache_global_to_local(
          stmt->src, CacheStatus::Read);
      auto local_load = std::make_unique<LocalLoadStmt>(alloca_stmt);
      stmt->replace_usages_with(local_load.get());
      modifier.insert_before(stmt, std::move(local_load));
      modifier.erase(stmt);
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (is_operand_loop_invariant_impl(stmt->dest, stmt->parent)) {
      auto alloca_stmt = cache_global_to_local(
          stmt->dest, CacheStatus::Write);
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
