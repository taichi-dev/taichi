#include "taichi/program/ir_bank.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/kernel.h"

TLANG_NAMESPACE_BEGIN

namespace {

uint64 hash(IRNode *stmt) {
  TI_ASSERT(stmt);
  // TODO: upgrade this using IR comparisons
  std::string serialized;
  irpass::re_id(stmt);
  irpass::print(stmt, &serialized);
  // TODO: separate kernel from IR template
  serialized += stmt->get_kernel()->name;
  uint64 ret = 0;
  for (uint64 i = 0; i < serialized.size(); i++) {
    ret = ret * 100000007UL + (uint64)serialized[i];
  }
  return ret;
}

}  // namespace

uint64 IRBank::get_hash(IRNode *ir) {
  auto result_iterator = hash_bank_.find(ir);
  if (result_iterator == hash_bank_.end()) {
    auto result = hash(ir);
    set_hash(ir, result);
    return result;
  }
  return result_iterator->second;
}

void IRBank::set_hash(IRNode *ir, uint64 hash) {
  hash_bank_[ir] = hash;
}

bool IRBank::insert(std::unique_ptr<IRNode> &&ir, uint64 hash) {
  IRHandle handle(ir.get(), hash);
  auto insert_place = ir_bank_.find(handle);
  if (insert_place == ir_bank_.end()) {
    ir_bank_.emplace(handle, std::move(ir));
    return true;
  }
  insert_to_trash_bin(std::move(ir));
  return false;
}

void IRBank::insert_to_trash_bin(std::unique_ptr<IRNode> &&ir) {
  trash_bin.push_back(std::move(ir));
}

IRNode *IRBank::find(IRHandle ir_handle) {
  auto result = ir_bank_.find(ir_handle);
  if (result == ir_bank_.end())
    return nullptr;
  return result->second.get();
}

IRHandle IRBank::fuse(IRHandle handle_a, IRHandle handle_b, Kernel *kernel) {
  auto &result = fuse_bank_[std::make_pair(handle_a, handle_b)];
  if (!result.empty()) {
    // assume the kernel is always the same when the ir handles are the same
    return result;
  }

  TI_TRACE("Begin uncached fusion");
  // We are about to change both |task_a| and |task_b|. Clone them first.
  auto cloned_task_a = handle_a.clone();
  auto cloned_task_b = handle_b.clone();
  auto task_a = cloned_task_a->as<OffloadedStmt>();
  auto task_b = cloned_task_b->as<OffloadedStmt>();
  // TODO: in certain cases this optimization can be wrong!
  // Fuse task b into task_a
  for (int j = 0; j < (int)task_b->body->size(); j++) {
    task_a->body->insert(std::move(task_b->body->statements[j]));
  }
  task_b->body->statements.clear();

  // replace all reference to the offloaded statement B to A
  irpass::replace_all_usages_with(task_a, task_b, task_a);

  irpass::full_simplify(task_a, /*after_lower_access=*/false, kernel);
  // For now, re_id is necessary for the hash to be correct.
  irpass::re_id(task_a);

  auto h = get_hash(task_a);
  result = IRHandle(task_a, h);
  insert(std::move(cloned_task_a), h);

  // TODO: since cloned_task_b->body is empty, can we remove this (i.e.,
  //  simply delete cloned_task_b here)?
  insert_to_trash_bin(std::move(cloned_task_b));

  return result;
}

// TODO: make this an IR pass
class ConstExprPropagation {
 public:
  static std::unordered_set<Stmt *> run(
      Block *block,
      const std::function<bool(Stmt *)> &is_const_seed) {
    std::unordered_set<Stmt *> const_stmts;

    auto is_const = [&](Stmt *stmt) {
      if (is_const_seed(stmt)) {
        return true;
      } else {
        return const_stmts.find(stmt) != const_stmts.end();
      }
    };

    for (auto &s : block->statements) {
      if (is_const(s.get())) {
        const_stmts.insert(s.get());
      } else if (auto binary = s->cast<BinaryOpStmt>()) {
        if (is_const(binary->lhs) && is_const(binary->rhs)) {
          const_stmts.insert(s.get());
        }
      } else if (auto unary = s->cast<UnaryOpStmt>()) {
        if (is_const(unary->operand)) {
          const_stmts.insert(s.get());
        }
      } else {
        // TODO: ...
      }
    }

    return const_stmts;
  }
};

IRHandle IRBank::demote_activation(IRHandle handle) {
  auto &result = demote_activation_bank_[handle];
  if (!result.empty()) {
    return result;
  }

  std::unique_ptr<IRNode> new_ir = handle.clone();

  OffloadedStmt *offload = new_ir->as<OffloadedStmt>();
  Block *body = offload->body.get();

  auto snode = offload->snode;
  TI_ASSERT(snode != nullptr);

  // TODO: for now we only deal with the top level. Is there an easy way to
  // extend this part?
  auto consts = ConstExprPropagation::run(body, [](Stmt *stmt) {
    if (stmt->is<ConstStmt>()) {
      return true;
    } else if (stmt->is<LoopIndexStmt>())
      return true;
    return false;
  });

  bool demoted = false;
  for (int k = 0; k < (int)body->statements.size(); k++) {
    Stmt *stmt = body->statements[k].get();
    if (auto ptr = stmt->cast<GlobalPtrStmt>(); ptr && ptr->activate) {
      bool can_demote = true;
      // TODO: test input mask?
      for (auto ind : ptr->indices) {
        if (consts.find(ind) == consts.end()) {
          // non-constant index
          can_demote = false;
        }
      }
      if (can_demote) {
        ptr->activate = false;
        demoted = true;
      }
    }
  }

  if (!demoted) {
    // Nothing demoted. Simply delete new_ir when this function returns.
    result = handle;
    return result;
  }

  result = IRHandle(new_ir.get(), get_hash(new_ir.get()));
  insert(std::move(new_ir), result.hash());
  return result;
}

std::pair<IRHandle, bool> IRBank::optimize_dse(
    IRHandle handle,
    const std::set<const SNode *> &snodes,
    bool verbose) {
  const OptimizeDseKey key(handle, snodes);
  auto &ret_handle = optimize_dse_bank_[key];
  if (!ret_handle.empty()) {
    // Already cached
    return std::make_pair(ret_handle, true);
  }

  std::unique_ptr<IRNode> new_ir = handle.clone();

  if (verbose) {
    TI_INFO("  DSE: before CFG");
    irpass::print(new_ir.get());
  }
  ControlFlowGraph::LiveVarAnalysisConfig lva_config;
  lva_config.eliminable_snodes = {snodes.begin(), snodes.end()};
  const bool modified = irpass::cfg_optimization(
      new_ir.get(), /*after_lower_access=*/false, lva_config);
  if (verbose) {
    TI_INFO("  DSE: after CFG, modified={}", modified);
    irpass::print(new_ir.get());
  }

  if (!modified) {
    // Nothing demoted. Simply delete new_ir when this function returns.
    ret_handle = handle;
    return std::make_pair(ret_handle, false);
  }

  ret_handle = IRHandle(new_ir.get(), get_hash(new_ir.get()));
  insert(std::move(new_ir), ret_handle.hash());
  return std::make_pair(ret_handle, false);
}

TLANG_NAMESPACE_END
