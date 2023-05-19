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
    std::vector<Stmt *> statements_to_delete;
    for (int i = 0; i <= (int)statements.size(); i++) {
      // TODO: in some cases BitStructStoreStmts across container statements can
      // still be merged, similar to basic block v.s. CFG optimizations.
      if (i == statements.size() || statements[i]->is_container_statement()) {
        for (const auto &[ptr, stores] : ptr_to_bit_struct_stores) {
          if (stores.size() == 1) {
            continue;
          }

          std::map<int, std::vector<Stmt *>> values;
          for (auto s : stores) {
            for (int j = 0; j < (int)s->ch_ids.size(); j++) {
              values[s->ch_ids[j]].push_back(s->values[j]);
            }
          }
          // Don't do store fusion when there's multiple stores to same child.
          // Example:
          //   <^qi4> $18 = get child [...] $17
          //   $19 : atomic bit_struct_store $17, ch_ids=[0], values=[$2]
          //   <i32> $20 = global load $18
          //   print "f[i]=", $20, "\n"
          //   <i32> $22 = add $20 $2
          //   $23 : atomic bit_struct_store $17, ch_ids=[0], values=[$22]
          // Here $19 and $23 can't be merged into a single store since the
          // result is used by `print`.
          // FIXME: Due to some hidden assumptions, the bug seems to be only
          // happen when there's multiple store to same child.
          bool multi_store_to_same_ch = false;
          for (auto const &[_, store_stmts] : values) {
            if (store_stmts.size() > 1) {
              multi_store_to_same_ch = true;
              break;
            }
          }
          if (multi_store_to_same_ch) {
            continue;
          }

          std::vector<int> ch_ids;
          std::vector<Stmt *> store_values;
          for (auto const &[ch_id, store_value_vec] : values) {
            ch_ids.push_back(ch_id);
            TI_ASSERT(store_value_vec.size() == 1);
            auto const &store_value = store_value_vec.front();
            store_values.push_back(store_value);
          }
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
        continue;
      }
      if (auto stmt = statements[i]->cast<BitStructStoreStmt>()) {
        ptr_to_bit_struct_stores[stmt->ptr].push_back(stmt);
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
