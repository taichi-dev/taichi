#include "taichi/analysis/bls_analyzer.h"

#include <memory>

#include "gtest/gtest.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/scratch_pad.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/transforms.h"
#include "taichi/struct/struct.h"
#include "tests/cpp/struct/fake_struct_compiler.h"

namespace taichi {
namespace lang {
namespace {

constexpr int kBlockSize = 8;

class BLSAnalyzerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<Index> indices = {Index{0}, Index{1}};
    root_snode_ = std::make_unique<SNode>(/*depth=*/0, /*t=*/SNodeType::root);
    parent_snode_ = &(root_snode_->dense(indices, /*sizes=*/kBlockSize));
    child_snode_ = &(parent_snode_->insert_children(SNodeType::place));
    child_snode_->dt = PrimitiveType::i32;

    FakeStructCompiler sc;
    sc.run(*root_snode_);

    for_stmt_ = std::make_unique<OffloadedStmt>(
        /*task_type=*/OffloadedTaskType::struct_for,
        /*arch=*/Arch::x64);
    for_stmt_->mem_access_opt.add_flag(child_snode_,
                                       SNodeAccessFlag::block_local);
    pads_.insert(child_snode_);

    builder_.set_insertion_point(
        {/*block=*/for_stmt_->body.get(), /*position=*/0});
  }

  std::unique_ptr<SNode> root_snode_{nullptr};
  SNode *parent_snode_{nullptr};
  SNode *child_snode_{nullptr};
  std::unique_ptr<OffloadedStmt> for_stmt_{nullptr};
  ScratchPads pads_;

  IRBuilder builder_;
};

TEST_F(BLSAnalyzerTest, Basic) {
  auto *loop_idx0_ = builder_.get_loop_index(for_stmt_.get(), /*index=*/0);
  auto *loop_idx1_ = builder_.get_loop_index(for_stmt_.get(), /*index=*/1);
  auto *c1 = builder_.get_int32(/*value=*/1);
  // x[i + 1, j]
  auto *idx = builder_.create_add(loop_idx0_, c1);
  auto *glb_ptr = builder_.create_global_ptr(child_snode_,
                                             /*indices=*/{idx, loop_idx1_});
  builder_.create_global_load(glb_ptr);
  // x[i, j - 3]
  auto *c3 = builder_.get_int32(/*value=*/3);
  idx = builder_.create_sub(loop_idx1_, c3);
  glb_ptr = builder_.create_global_ptr(child_snode_,
                                       /*indices=*/{loop_idx0_, idx});
  builder_.create_global_load(glb_ptr);

  BLSAnalyzer bls(for_stmt_.get(), &pads_);
  pads_.finalize();
  const auto &pad = pads_.get(child_snode_);
  EXPECT_EQ(pad.bounds.size(), 2);
  EXPECT_EQ(pad.bounds[0].low, 0);
  EXPECT_EQ(pad.bounds[0].high, 1 + kBlockSize);
  EXPECT_EQ(pad.bounds[1].low, -3);
  EXPECT_EQ(pad.bounds[1].high, kBlockSize);
}

}  // namespace
}  // namespace lang
}  // namespace taichi
