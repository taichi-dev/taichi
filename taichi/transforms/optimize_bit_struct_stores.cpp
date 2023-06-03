#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/pass.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/compile_config.h"

namespace {

using namespace taichi;
using namespace taichi::lang;

class CreateBitStructStores : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  CreateBitStructStores() {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
  }

  static void run(IRNode *root) {
    CreateBitStructStores pass;
    root->accept(&pass);
  }

  void visit(GlobalStoreStmt *stmt) override {
    auto get_ch = stmt->dest->cast<GetChStmt>();
    if (!get_ch || get_ch->input_snode->type != SNodeType::bit_struct)
      return;

    // We only handle bit_struct pointers here.

    auto s = Stmt::make<BitStructStoreStmt>(
        get_ch->input_ptr,
        std::vector<int>{get_ch->output_snode->id_in_bit_struct},
        std::vector<Stmt *>{stmt->val});
    stmt->replace_with(VecStatement(std::move(s)));
  }
};

class MergeBitStructStores : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  MergeBitStructStores() {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
  }

  static void run(IRNode *root) {
    while (true) {
      MergeBitStructStores pass;
      root->accept(&pass);
      if (!pass.modified_)
        break;
    }
  }

  void visit(Block *block) override {
    auto &statements = block->statements;
    std::unordered_map<Stmt *, std::vector<BitStructStoreStmt *>>
        ptr_to_bit_struct_stores;
    std::unordered_set<Stmt *> loaded_after_store;
    std::vector<Stmt *> statements_to_delete;
    for (int i = 0; i <= (int)statements.size(); i++) {
      // TODO: in some cases BitStructStoreStmts across container statements can
      // still be merged, similar to basic block v.s. CFG optimizations.
      if (i == statements.size() || statements[i]->is_container_statement()) {
        for (const auto &[ptr, stores] : ptr_to_bit_struct_stores) {
          if (stores.size() == 1) {
            continue;
          }

          std::vector<int> ch_ids;
          std::vector<Stmt *> store_values;
          for (auto s : stores) {
            for (int j = 0; j < (int)s->ch_ids.size(); j++) {
              auto const &ch_id = s->ch_ids[j];
              auto const &store_value = s->values[j];
              ch_ids.push_back(ch_id);
              store_values.push_back(store_value);
            }
          }

          auto ch_ids_dup = [&ch_ids = std::as_const(ch_ids)]() {
            std::unordered_set<int> ch_ids_set(ch_ids.begin(), ch_ids.end());
            return ch_ids_set.size() != ch_ids.size();
          };
          TI_ASSERT(!ch_ids_dup());
          // Now erase all (except the last) related BitSturctStoreStmts.
          // Replace the last one with a merged version.
          for (int j = 0; j < (int)stores.size() - 1; j++) {
            statements_to_delete.push_back(stores[j]);
          }
          stores.back()->replace_with(
              Stmt::make<BitStructStoreStmt>(ptr, ch_ids, store_values));
          modified_ = true;
        }
        ptr_to_bit_struct_stores.clear();
        loaded_after_store.clear();
      }
      // Skip bit store fusion when there's a load between multiple stores.
      // Example:
      //   <^qi16> $18 = get child [...] $17
      //   $178 : atomic bit_struct_store $17, ch_ids=[0], values=[$11]
      //   <i32> $20 = global load $18
      //   print "x[i]=", $20, "\n"
      //   <i32> $22 = add $11 $2
      //   <^qi16> $23 = get child [...] $17
      //   $179 : atomic bit_struct_store $17, ch_ids=[1], values=[$22]
      //   <i32> $25 = global load $23
      //   print "y[i]=", $25, "\n"
      // In this case, $178 and $179 cannot be merged into a single store
      // because the stored value $11 is loaded as $20 and then printed.
      else if (auto stmt = statements[i]->cast<BitStructStoreStmt>()) {
        // Phase 2: Find bit store after a marked load
        if (loaded_after_store.find(stmt->ptr) != loaded_after_store.end()) {
          // Disable store fusion for this bit struct
          ptr_to_bit_struct_stores.erase(stmt->ptr);
          loaded_after_store.erase(stmt->ptr);
        } else {
          ptr_to_bit_struct_stores[stmt->ptr].push_back(stmt);
        }
      } else if (auto load_stmt = statements[i]->cast<GlobalLoadStmt>()) {
        // Phase 1: Find and mark any global loads after bit_struct_store
        auto const &load_ops = load_stmt->src->get_operands();
        auto load_src = load_ops.empty() ? nullptr : load_ops.front();
        if (ptr_to_bit_struct_stores.find(load_src) !=
            ptr_to_bit_struct_stores.end()) {
          loaded_after_store.insert(load_src);
        }
      }
    }

    for (auto stmt : statements_to_delete) {
      block->erase(stmt);
    }

    for (auto &stmt : statements) {
      stmt->accept(this);
    }
  }

 private:
  bool modified_{false};
};

class DemoteAtomicBitStructStores : public BasicStmtVisitor {
 private:
  const std::unordered_map<OffloadedStmt *,
                           std::unordered_map<const SNode *, GlobalPtrStmt *>>
      &uniquely_accessed_bit_structs_;
  std::unordered_map<OffloadedStmt *,
                     std::unordered_map<const SNode *, GlobalPtrStmt *>>::
      const_iterator current_iterator_;
  bool modified_{false};

 public:
  using BasicStmtVisitor::visit;
  OffloadedStmt *current_offloaded;

  explicit DemoteAtomicBitStructStores(
      const std::unordered_map<
          OffloadedStmt *,
          std::unordered_map<const SNode *, GlobalPtrStmt *>>
          &uniquely_accessed_bit_structs)
      : uniquely_accessed_bit_structs_(uniquely_accessed_bit_structs),
        current_offloaded(nullptr) {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
  }

  void visit(BitStructStoreStmt *stmt) override {
    bool demote = false;
    TI_ASSERT(current_offloaded);
    if (current_offloaded->task_type == OffloadedTaskType::serial) {
      demote = true;
    } else if (current_offloaded->task_type == OffloadedTaskType::range_for ||
               current_offloaded->task_type == OffloadedTaskType::mesh_for ||
               current_offloaded->task_type == OffloadedTaskType::struct_for) {
      auto *snode = stmt->ptr->as<SNodeLookupStmt>()->snode;
      // Find the nearest non-bit-level ancestor
      while (snode->is_bit_level) {
        snode = snode->parent;
      }
      auto accessed_ptr_iterator = current_iterator_->second.find(snode);
      if (accessed_ptr_iterator != current_iterator_->second.end() &&
          accessed_ptr_iterator->second != nullptr) {
        demote = true;
      }
    }
    if (demote) {
      stmt->is_atomic = false;
      modified_ = true;
    }
  }

  void visit(OffloadedStmt *stmt) override {
    current_offloaded = stmt;
    if (stmt->task_type == OffloadedTaskType::range_for ||
        stmt->task_type == OffloadedTaskType::mesh_for ||
        stmt->task_type == OffloadedTaskType::struct_for) {
      current_iterator_ =
          uniquely_accessed_bit_structs_.find(current_offloaded);
    }
    // We don't need to visit TLS/BLS prologues/epilogues.
    if (stmt->body) {
      stmt->body->accept(this);
    }
    current_offloaded = nullptr;
  }

  static bool run(IRNode *node,
                  const std::unordered_map<
                      OffloadedStmt *,
                      std::unordered_map<const SNode *, GlobalPtrStmt *>>
                      &uniquely_accessed_bit_structs) {
    DemoteAtomicBitStructStores demoter(uniquely_accessed_bit_structs);
    node->accept(&demoter);
    return demoter.modified_;
  }
};

}  // namespace

namespace taichi::lang {

namespace irpass {
void optimize_bit_struct_stores(IRNode *root,
                                const CompileConfig &config,
                                AnalysisManager *amgr) {
  TI_AUTO_PROF;
  CreateBitStructStores::run(root);
  die(root);  // remove unused GetCh
  if (config.quant_opt_store_fusion) {
    MergeBitStructStores::run(root);
  }
  if (config.quant_opt_atomic_demotion) {
    auto *res = amgr->get_pass_result<GatherUniquelyAccessedBitStructsPass>();
    TI_ASSERT_INFO(res,
                   "The optimize_bit_struct_stores pass must be after the "
                   "gather_uniquely_accessed_bit_structs pass when "
                   "config.quant_opt_atomic_demotion is true.");
    DemoteAtomicBitStructStores::run(root, res->uniquely_accessed_bit_structs);
  }
}

}  // namespace irpass

}  // namespace taichi::lang
