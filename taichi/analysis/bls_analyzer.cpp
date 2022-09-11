#include "taichi/analysis/bls_analyzer.h"

#include "taichi/system/profiler.h"
#include "taichi/ir/analysis.h"

namespace taichi {
namespace lang {

BLSAnalyzer::BLSAnalyzer(OffloadedStmt *for_stmt, ScratchPads *pads)
    : for_stmt_(for_stmt), pads_(pads) {
  TI_AUTO_PROF;
  allow_undefined_visitor = true;
  invoke_default_visitor = false;
  for (auto &snode : for_stmt->mem_access_opt.get_snodes_with_flag(
           SNodeAccessFlag::block_local)) {
    auto *block = snode->parent;
    if (block_indices_.find(block) == block_indices_.end()) {
      generate_block_indices(block, &block_indices_[block]);
    }
  }
}

// static
void BLSAnalyzer::generate_block_indices(SNode *snode, BlockIndices *indices) {
  // NOTE: Assuming not vectorized
  for (int i = 0; i < snode->num_active_indices; i++) {
    auto j = snode->physical_index_position[i];
    indices->push_back({/*low=*/0, /*high=*/snode->extractors[j].shape - 1});
  }
}

void BLSAnalyzer::record_access(Stmt *stmt, AccessFlag flag) {
  if (!analysis_ok_) {
    return;
  }
  if (!stmt->is<GlobalPtrStmt>())
    return;  // local alloca
  auto ptr = stmt->as<GlobalPtrStmt>();
  auto snode = ptr->snode;
  if (!pads_->has(snode)) {
    return;
  }
  bool matching_indices = true;
  std::vector<IndexRange> offsets;
  std::vector<int> coeffs;
  offsets.resize(ptr->indices.size());
  coeffs.resize(ptr->indices.size());
  const int num_indices = (int)ptr->indices.size();
  for (int i = 0; i < num_indices; i++) {
    auto diff =
        irpass::analysis::value_diff_loop_index(ptr->indices[i], for_stmt_, i);
    if (diff.related() && diff.coeff > 0) {
      offsets[i].low = diff.low;
      offsets[i].high = diff.high;
      coeffs[i] = diff.coeff;
    } else {
      matching_indices = false;
      analysis_ok_ = false;
    }
  }
  if (matching_indices) {
    auto *block = snode->parent;
    const auto &index_bounds = block_indices_[block];
    std::vector<int> index(num_indices, 0);
    std::function<void(int)> visit = [&](int dimension) {
      if (dimension == num_indices) {
        pads_->access(snode, coeffs, index, flag);
        return;
      }
      for (int i = (index_bounds[dimension].low + offsets[dimension].low);
           i < (index_bounds[dimension].high + offsets[dimension].high); i++) {
        index[dimension] = i;
        visit(dimension + 1);
      }
    };
    visit(0);
  }
}

// Do not eliminate global data access
void BLSAnalyzer::visit(GlobalLoadStmt *stmt) {
  record_access(stmt->src, AccessFlag::read);
}

void BLSAnalyzer::visit(GlobalStoreStmt *stmt) {
  record_access(stmt->dest, AccessFlag::write);
}

void BLSAnalyzer::visit(AtomicOpStmt *stmt) {
  if (stmt->op_type == AtomicOpType::add) {
    record_access(stmt->dest, AccessFlag::accumulate);
  }
}

void BLSAnalyzer::visit(Stmt *stmt) {
  TI_ASSERT(!stmt->is_container_statement());
}

bool BLSAnalyzer::run() {
  const auto &block = for_stmt_->body;

  for (int i = 0; i < (int)block->statements.size(); i++) {
    block->statements[i]->accept(this);
  }

  return analysis_ok_;
}

}  // namespace lang
}  // namespace taichi
