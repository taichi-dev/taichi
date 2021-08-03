#include <memory>

#include "gtest/gtest.h"
#include "taichi/analysis/arithmetic_interpretor.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/transforms/scalar_pointer_lowerer.h"
#include "tests/cpp/struct/fake_struct_compiler.h"

namespace taichi {
namespace lang {
namespace {

constexpr int kPointerSize = 4;
constexpr int kDenseSize = 8;

class LowererImpl : public ScalarPointerLowerer {
 public:
  using ScalarPointerLowerer::ScalarPointerLowerer;
  std::vector<LinearizeStmt *> linears;

 protected:
  Stmt *handle_snode_at_level(int level,
                              LinearizeStmt *linearized,
                              Stmt *last) override {
    linears.push_back(linearized);
    return last;
  }
};

class ScalarPointerLowererTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_snode_ = std::make_unique<SNode>(/*depth=*/0, /*t=*/SNodeType::root);
    const std::vector<Axis> axes = {Axis{0}};
    ptr_snode_ = &(root_snode_->pointer(axes, kPointerSize, false));
    dense_snode_ = &(ptr_snode_->dense(axes, kDenseSize, false));
    // Must end with a `place` SNode.
    leaf_snode_ = &(dense_snode_->insert_children(SNodeType::place));
    leaf_snode_->dt = PrimitiveType::f32;

    FakeStructCompiler sc;
    sc.run(*root_snode_);
  }

  const CompileConfig cfg_;
  std::unique_ptr<SNode> root_snode_{nullptr};
  SNode *ptr_snode_{nullptr};
  SNode *dense_snode_{nullptr};
  SNode *leaf_snode_{nullptr};
};

TEST_F(ScalarPointerLowererTest, Basic) {
  IRBuilder builder;
  for (int i = 0; i < kPointerSize; ++i) {
    for (int j = 0; j < kDenseSize; ++j) {
      const int loop_index = (i * kDenseSize) + j;
      VecStatement lowered;
      LowererImpl lowerer{leaf_snode_,
                          std::vector<Stmt *>{builder.get_int32(loop_index)},
                          SNodeOpType::undefined,
                          /*is_bit_vectorized=*/false,
                          &lowered,
                          /*packed=*/false};
      lowerer.run();
      // There are three linearized stmts:
      // 0: for root
      // 1: for pointer
      // 2: for dense
      constexpr int kPointerLevel = 1;
      constexpr int kDenseLevel = 2;
      ASSERT_EQ(lowerer.linears.size(), 3);

      auto block = builder.extract_ir();
      block->insert(std::move(lowered));
      // Set types so that ArithmeticInterpretor can run correctly
      irpass::type_check(block.get(), cfg_);

      ArithmeticInterpretor::CodeRegion code_region;
      code_region.block = block.get();

      ArithmeticInterpretor::EvalContext init_ctx;
      for (auto &stmt : code_region.block->statements) {
        if (stmt->is<GetRootStmt>()) {
          init_ctx.ignore(stmt.get());
          break;
        }
      }

      ArithmeticInterpretor ai;
      code_region.end = lowerer.linears[kPointerLevel];
      auto res_opt = ai.evaluate(code_region, init_ctx);
      ASSERT_TRUE(res_opt.has_value());
      EXPECT_EQ(res_opt.value(), i);

      code_region.end = lowerer.linears[kDenseLevel];
      res_opt = ai.evaluate(code_region, init_ctx);
      ASSERT_TRUE(res_opt.has_value());
      EXPECT_EQ(res_opt.value(), j);
    }
  }
}

}  // namespace
}  // namespace lang
}  // namespace taichi
