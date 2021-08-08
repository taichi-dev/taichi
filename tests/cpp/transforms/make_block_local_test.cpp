#include <memory>

#include "gtest/gtest.h"
#include "taichi/analysis/arithmetic_interpretor.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/struct/struct.h"
#include "tests/cpp/struct/fake_struct_compiler.h"

namespace taichi {
namespace lang {
namespace {

class MakeBlockLocalTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_snode_ = std::make_unique<SNode>(/*depth=*/0, /*t=*/SNodeType::root);
  }

  void initialize(int pointer_size, int block_size) {
    // The SNode tree of this test looks like the following
    //
    // |root_snode_|  // ti.root
    // \
    //  |
    //  +- |pointer_snode_|  // .pointer(ti.ij, |pointer_size|)
    //     \
    //      |
    //      +- |bls_snode_|  // .dense(ti.ij, |block_size|s)
    //      |
    //      +- |struct_for_snode_|  // .dynamic(ti.k, ...)
    //
    // The |struct_for_snode_| is the one being iterated over in the struct-for
    // offloaded task |for_stmt_|, meaning that the loop index is based on this
    // SNode. On the other hand, |bls_snode_| is the one we want to cache into
    // the BLS buffer (shared memory).
    //
    //
    // |bls_snode_| has a larger shape than |struct_for_snode_|, because we
    // want to see if the tests can handle the loop index scaling multiplier
    // (block_size) and infer the BLS size correctly.
    const std::vector<Axis> axes = {Axis{0}, Axis{1}};
    pointer_snode_ = &(root_snode_->pointer(axes, pointer_size, false));

    bls_snode_ = &(pointer_snode_->dense(axes, /*sizes=*/block_size, false));
    bls_place_snode_ = &(bls_snode_->insert_children(SNodeType::place));
    bls_place_snode_->dt = PrimitiveType::f32;

    struct_for_snode_ = &(pointer_snode_->dynamic({Axis{2}}, /*n=*/1024,
                                                  /*chunk_size=*/128, false));
    struct_for_place_snode_ =
        &(struct_for_snode_->insert_children(SNodeType::place));
    struct_for_place_snode_->dt = PrimitiveType::i32;

    FakeStructCompiler sc;
    sc.run(*root_snode_);

    for_stmt_ = std::make_unique<OffloadedStmt>(
        /*task_type=*/OffloadedTaskType::struct_for,
        /*arch=*/Arch::x64);
    for_stmt_->mem_access_opt.add_flag(bls_place_snode_,
                                       SNodeAccessFlag::block_local);
    for_stmt_->snode = struct_for_place_snode_;
    for_stmt_->block_dim = 64;

    builder_.set_insertion_point(
        {/*block=*/for_stmt_->body.get(), /*position=*/0});
  }

  int get_block_corner(int loop_index) const {
    return loop_index;
  }

  int get_block_size(int axis) const {
    axis = bls_snode_->physical_index_position[axis];
    return (1 << bls_snode_->extractors[axis].num_bits);
  }

  std::unique_ptr<SNode> root_snode_{nullptr};
  SNode *pointer_snode_{nullptr};
  SNode *bls_snode_{nullptr};
  SNode *bls_place_snode_{nullptr};
  SNode *struct_for_snode_{nullptr};
  SNode *struct_for_place_snode_{nullptr};
  std::unique_ptr<OffloadedStmt> for_stmt_{nullptr};

  IRBuilder builder_;
};

TEST_F(MakeBlockLocalTest, Basic) {
  initialize(/*pointer_size=*/2, /*block_size=*/4);

  // This test has two global loads to |bls_place_snode_|:
  // * [block_size * i, block_size * j]
  // * [block_size * i - 1, block_size * j - 3]
  //
  // So the block size at ti.i or ti.j is 4, respectively.
  // The BLS pad size at ti.i = (4 + 0 - (-1)) = 5, ti.j = (4 + 0 - (-3)) = 7.
  auto *loop_idx0_ = builder_.get_loop_index(for_stmt_.get(), /*index=*/0);
  auto *loop_idx1_ = builder_.get_loop_index(for_stmt_.get(), /*index=*/1);

  // x[block_size * i, block_size * j]
  auto *b0 = builder_.get_int32(/*value=*/get_block_size(0));
  auto *b1 = builder_.get_int32(/*value=*/get_block_size(1));
  auto *idx0 = builder_.create_mul(loop_idx0_, b0);
  auto *idx1 = builder_.create_mul(loop_idx1_, b1);
  auto *glb_ptr = builder_.create_global_ptr(bls_place_snode_,
                                             /*indices=*/{idx0, idx1});
  builder_.create_global_load(glb_ptr);

  // x[block_size * i - 1, block_size * j - 3]
  constexpr int kNeg1 = -1;
  constexpr int kNeg3 = -3;
  auto *c0 = builder_.get_int32(/*value=*/kNeg1);
  auto *c1 = builder_.get_int32(/*value=*/kNeg3);
  idx0 = builder_.create_add(idx0, c0);
  idx1 = builder_.create_add(idx1, c1);
  glb_ptr = builder_.create_global_ptr(bls_place_snode_,
                                       /*indices=*/{idx0, idx1});
  builder_.create_global_load(glb_ptr);
  irpass::make_block_local(for_stmt_.get(), CompileConfig{},
                           MakeBlockLocalPass::Args{});

  auto loop_shape_at = [p = struct_for_place_snode_](int axis) {
    return p->shape_along_axis(axis);
  };
  // Runs over all the indices covered by |struct_for_place_snode_|. Checks if
  // the generated BLS offset is correct for the given global loop indices.
  for (int idx0 = 0; idx0 < loop_shape_at(0); ++idx0) {
    for (int idx1 = 0; idx1 < loop_shape_at(1); ++idx1) {
      const int loop_indices_vals[] = {idx0, idx1};
      const int block_corner_vals[] = {
          get_block_corner(idx0),
          get_block_corner(idx1),
      };
      int expected_bls_offset_in_bytes =
          (get_block_size(0) * (loop_indices_vals[0] - block_corner_vals[0]) -
           /*bls_bounds[0].lower=*/kNeg1);
      expected_bls_offset_in_bytes *=
          /*bls_stride[0]=*/(get_block_size(/*axis=*/1) - kNeg3);
      expected_bls_offset_in_bytes +=
          (get_block_size(1) * (loop_indices_vals[1] - block_corner_vals[1]) -
           /*bls_bounds[1].lower=*/kNeg3);
      expected_bls_offset_in_bytes *= sizeof(float);

      ArithmeticInterpretor::CodeRegion code_region;
      ArithmeticInterpretor::EvalContext init_ctx;
      code_region.block = for_stmt_->body.get();

      {
        // Build ArithemticInterpretor's initial context
        const auto &stmts = code_region.block->statements;
        code_region.begin = stmts.front().get();
        // The body of |for_stmt_| roughly looks like this:
        //
        // <i32> $217 = loop $216 index 0
        // <i32> $218 = loop $216 index 1
        // <i32> $255 = loop $216 block corner index 0
        // ... a bunch of arithmetic ops
        // <i32> $261 = loop $216 block corner index 1
        // ... a bunch of arithmetic ops
        // <i32> $271 = add $269 $270
        // <*f32> $272 = block local ptr (offset = $271)
        // ... other instructions that are irrelevant

        for (int i = 0; i < stmts.size(); ++i) {
          const auto *stmt = stmts[i].get();
          if (stmt->is<LoopIndexStmt>()) {
            const auto *li = stmt->as<LoopIndexStmt>();
            init_ctx.insert(stmt, TypedConstant(PrimitiveType::i32,
                                                loop_indices_vals[li->index]));
          } else if (stmt->is<BlockCornerIndexStmt>()) {
            const auto *bc = stmt->as<BlockCornerIndexStmt>();
            init_ctx.insert(stmt, TypedConstant(PrimitiveType::i32,
                                                block_corner_vals[bc->index]));
          } else if (stmt->is<BlockLocalPtrStmt>()) {
            code_region.end = stmts[i].get();
            break;
          }
        }
      }

      ArithmeticInterpretor ai;
      auto bls_offset_opt = ai.evaluate(code_region, init_ctx);
      ASSERT_TRUE(bls_offset_opt.has_value());
      EXPECT_EQ(bls_offset_opt.value().val_int32(),
                expected_bls_offset_in_bytes);
    }
  }
}

}  // namespace
}  // namespace lang
}  // namespace taichi
