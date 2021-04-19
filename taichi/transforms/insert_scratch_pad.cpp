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

  // The lowest and highest index in each dimension.
  using BlockIndices = std::vector<std::pair<int, int>>;

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
        generate_block_indices(block, &block_indices[block]);
      }
    }

    const auto &block = for_stmt->body;

    for (int i = 0; i < (int)block->statements.size(); i++) {
      block->statements[i]->accept(this);
    }
  }

  // Generate the index bounds in a SNode (block). E.g., a dense(ti.ij, (2, 4))
  // SNode has index bounds [[0, 1], [0, 3]].
  static void generate_block_indices(SNode *snode, BlockIndices *indices) {
    // NOTE: Assuming not vectorized
    for (int i = 0; i < snode->num_active_indices; i++) {
      auto j = snode->physical_index_position[i];
      indices->emplace_back(0, (1 << snode->extractors[j].num_bits) - 1);
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
        const auto &index_bounds = block_indices[block];
        std::vector<int> index(num_indices, 0);
        std::function<void(int)> visit = [&](int dimension) {
          if (dimension == num_indices) {
            pads->access(snode, index, flag);
            return;
          }
          for (int i = index_bounds[dimension].first + offsets[dimension].first;
               i < index_bounds[dimension].second + offsets[dimension].second;
               i++) {
            index[dimension] = i;
            visit(dimension + 1);
          }
        };
        visit(0);
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
