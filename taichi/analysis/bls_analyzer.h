#pragma once

#include "taichi/ir/visitors.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/scratch_pad.h"

namespace taichi {
namespace lang {

// Figure out accessed SNodes, and their ranges in this for stmt
class BLSAnalyzer : public BasicStmtVisitor {
  using BasicStmtVisitor::visit;

 public:
  // The lowest and highest index in each dimension.
  struct IndexRange {
    int low{0};
    int high{0};
  };
  using BlockIndices = std::vector<IndexRange>;

  BLSAnalyzer(OffloadedStmt *for_stmt, ScratchPads *pads);

  void visit(GlobalPtrStmt *stmt) override {
  }

  // Do not eliminate global data access
  void visit(GlobalLoadStmt *stmt) override;

  void visit(GlobalStoreStmt *stmt) override;

  void visit(AtomicOpStmt *stmt) override;

  void visit(Stmt *stmt) override;

  /**
   * Run the block local analysis
   * @return: true if the block range could be successfully inferred
   */
  bool run();

 private:
  // Generate the index bounds in a SNode (block). E.g., a dense(ti.ij, (2, 4))
  // SNode has index bounds [[0, 1], [0, 3]].
  static void generate_block_indices(SNode *snode, BlockIndices *indices);

  void record_access(Stmt *stmt, AccessFlag flag);

  OffloadedStmt *for_stmt_{nullptr};
  ScratchPads *pads_{nullptr};
  std::unordered_map<SNode *, BlockIndices> block_indices_;
  // true means analysis is OK for now
  // it could be failed by any of the following reasons
  // compiler could not infer the scratch pad range at compile time
  // ...
  bool analysis_ok_{true};
};

}  // namespace lang
}  // namespace taichi
