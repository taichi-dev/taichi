#include "taichi/transforms/loop_invariant_detector.h"
#include "taichi/ir/analysis.h"

namespace taichi::lang {

class CacheLoopInvariantGlobalVars : public LoopInvariantDetector {
 public:
  using LoopInvariantDetector::visit;

  enum class CacheStatus {
    None = 0,
    Read = 1,
    Write = 2,
    ReadWrite = 3,
  };

  typedef std::unordered_map<Stmt *, std::pair<CacheStatus, AllocaStmt *>>
      CacheMap;
  std::vector<CacheMap> cached_maps;

  DelayedIRModifier modifier;
  std::unordered_map<const SNode *, GlobalPtrStmt *> loop_unique_ptr_;
  std::unordered_map<std::vector<int>,
                     ExternalPtrStmt *,
                     hashing::Hasher<std::vector<int>>>
      loop_unique_arr_ptr_;
  std::unordered_set<MatrixPtrStmt *> loop_unique_matrix_ptr_;

  OffloadedStmt *current_offloaded;

  explicit CacheLoopInvariantGlobalVars(const CompileConfig &config)
      : LoopInvariantDetector(config) {
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->task_type == OffloadedTaskType::range_for ||
        stmt->task_type == OffloadedTaskType::mesh_for ||
        stmt->task_type == OffloadedTaskType::struct_for) {
      auto uniquely_accessed_pointers =
          irpass::analysis::gather_uniquely_accessed_pointers(stmt);
      loop_unique_ptr_ = std::move(std::get<0>(uniquely_accessed_pointers));
      loop_unique_arr_ptr_ = std::move(std::get<1>(uniquely_accessed_pointers));
      loop_unique_matrix_ptr_ =
          std::move(std::get<2>(uniquely_accessed_pointers));
    }
    current_offloaded = stmt;
    // We don't need to visit TLS/BLS prologues/epilogues.
    if (stmt->body) {
      if (stmt->task_type == OffloadedStmt::TaskType::range_for ||
          stmt->task_type == OffloadedTaskType::mesh_for ||
          stmt->task_type == OffloadedStmt::TaskType::struct_for)
        visit_loop(stmt->body.get());
      else
        stmt->body->accept(this);
    }
    current_offloaded = nullptr;
  }

  bool is_offload_unique(Stmt *stmt) {
    if (current_offloaded->task_type == OffloadedTaskType::serial) {
      return true;
    }

    // Handle GlobalPtrStmt
    bool is_global_ptr_stmt = false;
    GlobalPtrStmt *global_ptr = nullptr;
    if (stmt->is<GlobalPtrStmt>()) {
      is_global_ptr_stmt = true;
      global_ptr = stmt->as<GlobalPtrStmt>();
    } else if (stmt->is<MatrixPtrStmt>() &&
               stmt->as<MatrixPtrStmt>()->origin->is<GlobalPtrStmt>()) {
      if (loop_unique_matrix_ptr_.find(stmt->as<MatrixPtrStmt>()) ==
          loop_unique_matrix_ptr_.end()) {
        return false;
      }

      is_global_ptr_stmt = true;
      global_ptr = stmt->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    }

    if (global_ptr) {
      auto snode = global_ptr->snode;
      if (loop_unique_ptr_[snode] == nullptr ||
          loop_unique_ptr_[snode]->indices.empty()) {
        // not uniquely accessed
        return false;
      }
      if (current_offloaded->mem_access_opt.has_flag(
              snode, SNodeAccessFlag::block_local) ||
          current_offloaded->mem_access_opt.has_flag(
              snode, SNodeAccessFlag::mesh_local)) {
        // BLS does not support write access yet so we keep atomic_adds.
        return false;
      }
      return true;
    }

    // Handle ExternalPtrStmt
    bool is_external_ptr_stmt = false;
    ExternalPtrStmt *dest_ptr = nullptr;
    if (stmt->is<ExternalPtrStmt>()) {
      is_external_ptr_stmt = true;
      dest_ptr = stmt->as<ExternalPtrStmt>();
    } else if (stmt->is<MatrixPtrStmt>() &&
               stmt->as<MatrixPtrStmt>()->origin->is<ExternalPtrStmt>()) {
      if (loop_unique_matrix_ptr_.find(stmt->as<MatrixPtrStmt>()) ==
          loop_unique_matrix_ptr_.end()) {
        return false;
      }

      is_external_ptr_stmt = true;
      dest_ptr = stmt->as<MatrixPtrStmt>()->origin->as<ExternalPtrStmt>();
    }

    if (is_external_ptr_stmt) {
      if (dest_ptr->indices.empty()) {
        return false;
      }
      ArgLoadStmt *arg_load_stmt = dest_ptr->base_ptr->as<ArgLoadStmt>();
      std::vector<int> arg_id = arg_load_stmt->arg_id;
      if (loop_unique_arr_ptr_[arg_id] == nullptr) {
        // Not loop unique
        return false;
      }
      return true;
      // TODO: Is BLS / Mem Access Opt a thing for any_arr?
    }
    return false;
  }

  void visit_loop(Block *body) override {
    cached_maps.emplace_back();
    LoopInvariantDetector::visit_loop(body);
    cached_maps.pop_back();
  }

  void add_writeback(AllocaStmt *alloca_stmt, Stmt *global_var, int depth) {
    auto final_value = std::make_unique<LocalLoadStmt>(alloca_stmt);
    auto global_store =
        std::make_unique<GlobalStoreStmt>(global_var, final_value.get());
    modifier.insert_after(get_loop_stmt(depth), std::move(global_store));
    modifier.insert_after(get_loop_stmt(depth), std::move(final_value));
  }

  void set_init_value(AllocaStmt *alloca_stmt, Stmt *global_var, int depth) {
    auto new_global_load = std::make_unique<GlobalLoadStmt>(global_var);
    auto local_store =
        std::make_unique<LocalStoreStmt>(alloca_stmt, new_global_load.get());
    modifier.insert_before(get_loop_stmt(depth), std::move(new_global_load));
    modifier.insert_before(get_loop_stmt(depth), std::move(local_store));
  }

  AllocaStmt *cache_global_to_local(Stmt *dest, CacheStatus status, int depth) {
    if (auto &[cached_status, alloca_stmt] = cached_maps[depth][dest];
        cached_status != CacheStatus::None) {
      // The global variable has already been cached.
      if (cached_status == CacheStatus::Read && status == CacheStatus::Write) {
        add_writeback(alloca_stmt, dest, depth);
        cached_status = CacheStatus::ReadWrite;
      }
      return alloca_stmt;
    }
    auto alloca_unique =
        std::make_unique<AllocaStmt>(dest->ret_type.ptr_removed());
    auto alloca_stmt = alloca_unique.get();
    modifier.insert_before(get_loop_stmt(depth), std::move(alloca_unique));
    set_init_value(alloca_stmt, dest, depth);
    if (status == CacheStatus::Write) {
      add_writeback(alloca_stmt, dest, depth);
    }
    cached_maps[depth][dest] = {status, alloca_stmt};
    return alloca_stmt;
  }

  std::optional<int> find_cache_depth_if_cacheable(Stmt *operand,
                                                   Block *current_scope) {
    if (!is_offload_unique(operand)) {
      return std::nullopt;
    }
    std::optional<int> depth;
    for (int n = loop_blocks.size() - 1; n > 0; n--) {
      if (is_operand_loop_invariant(operand, current_scope, n)) {
        depth = n;
      } else {
        break;
      }
    }
    return depth;
  }

  void visit(GlobalLoadStmt *stmt) override {
    if (auto depth = find_cache_depth_if_cacheable(stmt->src, stmt->parent)) {
      auto alloca_stmt =
          cache_global_to_local(stmt->src, CacheStatus::Read, depth.value());
      auto local_load = std::make_unique<LocalLoadStmt>(alloca_stmt);
      stmt->replace_usages_with(local_load.get());
      modifier.insert_before(stmt, std::move(local_load));
      modifier.erase(stmt);
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (auto depth = find_cache_depth_if_cacheable(stmt->dest, stmt->parent)) {
      auto alloca_stmt =
          cache_global_to_local(stmt->dest, CacheStatus::Write, depth.value());
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
}  // namespace taichi::lang
