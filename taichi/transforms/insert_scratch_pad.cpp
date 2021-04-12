#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/scratch_pad.h"
#include "taichi/system/profiler.h"

TLANG_NAMESPACE_BEGIN

// TODO: rename scratch_pad to block_local_cache? Need to get rid of the
// scratch_pad term

// Figure out accessed SNodes, and their ranges in this for stmt
class BLSAnalysis : public BasicStmtVisitor {
  using BasicStmtVisitor::visit;

 public:
  OffloadedStmt *for_stmt;
  ScratchPads *pads;

  using BlockIndices = std::vector<std::vector<int>>;

  std::unordered_map<SNode *, BlockIndices> block_indices;

  BLSAnalysis(OffloadedStmt *for_stmt, ScratchPads *pads)
      : for_stmt(for_stmt), pads(pads) {
    TI_AUTO_PROF;
    allow_undefined_visitor = true;
    invoke_default_visitor = false;

    for (auto &snode : for_stmt->mem_access_opt.get_snodes_with_flag(
             SNodeAccessFlag::block_local)) {
      auto block = snode->parent;
      if (block_indices.find(block) == block_indices.end()) {
        generate_block_indices(block, block_indices[block], {}, 0);
      }
    }

    const auto &block = for_stmt->body;

    for (int i = 0; i < (int)block->statements.size(); i++) {
      block->statements[i]->accept(this);
    }
  }

  // Recursively (dimension by dimension) generate the indices in a SNode
  // (block). E.g., a dense(ti.ij, (2, 4)) SNode has indices
  // [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
  void generate_block_indices(SNode *snode,
                              BlockIndices &block_indices,
                              std::vector<int> index,
                              int index_id) {
    // NOTE: Assuming not vectorized
    if (index_id == taichi_max_num_indices) {
      block_indices.push_back(index);
      return;
    }

    if (snode->extractors[index_id].active) {
      for (int i = 0; i < (1 << snode->extractors[index_id].num_bits); i++) {
        auto new_index = index;
        new_index.push_back(i);
        generate_block_indices(snode, block_indices, new_index, index_id + 1);
      }
    } else {
      generate_block_indices(snode, block_indices, index, index_id + 1);
    }
  }

  void visit(GlobalPtrStmt *stmt) override {
  }

  void record_access(Stmt *stmt, AccessFlag flag) {
    if (!stmt->is<GlobalPtrStmt>())
      return;  // local alloca
    auto ptr = stmt->as<GlobalPtrStmt>();
    for (int l = 0; l < stmt->width(); l++) {
      auto snode = ptr->snodes[l];
      if (!pads->has(snode)) {
        continue;
      }
      bool matching_indices = true;
      std::vector<std::pair<int, int>> offsets;
      offsets.resize(ptr->indices.size());
      int num_indices = (int)ptr->indices.size();
      for (int i = 0; i < num_indices; i++) {
        auto diff = irpass::analysis::value_diff_loop_index(ptr->indices[i],
                                                            for_stmt, i);
        if (diff.linear_related()) {
          offsets[i].first = diff.low;
          offsets[i].second = diff.high;
        } else {
          matching_indices = false;
        }
      }
      if (matching_indices) {
        auto block = snode->parent;
        for (const auto &bind : block_indices[block]) {
          std::function<void(std::vector<int>, int)> visit =
              [&](std::vector<int> ind, int depth) {
                if (depth == num_indices) {
                  pads->access(snode, ind, flag);
                  return;
                }
                for (int i = offsets[depth].first; i < offsets[depth].second;
                     i++) {
                  ind[depth] = bind[depth] + i;
                  visit(ind, depth + 1);
                }
              };
          visit(bind, 0);
        }
      }
    }
  }

  // Do not eliminate global data access
  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);  // TODO: support vectorization
    record_access(stmt->src, AccessFlag::read);
  }

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);  // TODO: support vectorization
    record_access(stmt->dest, AccessFlag::write);
  }

  void visit(AtomicOpStmt *stmt) override {
    if (stmt->op_type == AtomicOpType::add) {
      record_access(stmt->dest, AccessFlag::accumulate);
    }
  }

  void visit(Stmt *stmt) override {
    TI_ASSERT(!stmt->is_container_statement());
  }
};

namespace irpass {

std::unique_ptr<ScratchPads> initialize_scratch_pad(OffloadedStmt *offload) {
  TI_AUTO_PROF
  TI_ASSERT(offload->task_type == OffloadedTaskType::struct_for);
  std::unique_ptr<ScratchPads> pads;
  pads = std::make_unique<ScratchPads>();
  for (auto snode : offload->mem_access_opt.get_snodes_with_flag(
           SNodeAccessFlag::block_local)) {
    pads->insert(snode);
  }
  BLSAnalysis _(offload, pads.get());
  pads->finalize();
  return pads;
}

}  // namespace irpass

TLANG_NAMESPACE_END
