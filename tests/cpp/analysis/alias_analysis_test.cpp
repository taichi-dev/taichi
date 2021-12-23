#include <algorithm>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"

namespace taichi {
namespace lang {
namespace irpass {
namespace analysis {
namespace {

std::vector<Stmt *> make_const_indices(const std::vector<int> indices,
                                       IRBuilder *builder) {
  std::vector<Stmt *> result;
  result.reserve(indices.size());
  std::for_each(indices.begin(), indices.end(), [builder, &result](int i) {
    result.push_back(builder->get_int32(i));
  });
  return result;
}

}  // namespace

TEST(AliasAnalysis, Alloca) {
  IRBuilder builder;

  auto *alloca1 = builder.create_local_var(PrimitiveType::f32);
  auto *alloca2 = builder.create_local_var(PrimitiveType::f32);

  auto aa = alias_analysis(alloca1, alloca1);
  EXPECT_EQ(aa, AliasResult::same);
  aa = alias_analysis(alloca1, alloca2);
  EXPECT_EQ(aa, AliasResult::different);
}

TEST(AliasAnalysis, GlobalPtr_Same) {
  auto snode = std::make_unique<SNode>(/*depth=*/1, /*t=*/SNodeType::place);

  IRBuilder builder;
  const auto indices = make_const_indices({1, 2, 3}, &builder);
  auto *gptr1 = builder.create_global_ptr(snode.get(), indices);
  auto *gptr2 = builder.create_global_ptr(snode.get(), indices);

  const auto aa = alias_analysis(gptr1, gptr2);
  EXPECT_EQ(aa, AliasResult::same);
}

TEST(AliasAnalysis, GlobalPtr_DiffIndices) {
  auto snode = std::make_unique<SNode>(/*depth=*/1, /*t=*/SNodeType::place);

  IRBuilder builder;
  const auto indices1 = make_const_indices({1, 2, 3}, &builder);
  const auto indices2 = make_const_indices({1, 3, 2}, &builder);
  auto *gptr1 = builder.create_global_ptr(snode.get(), indices1);
  auto *gptr2 = builder.create_global_ptr(snode.get(), indices2);

  const auto aa = alias_analysis(gptr1, gptr2);
  EXPECT_EQ(aa, AliasResult::different);
}

TEST(AliasAnalysis, GlobalPtr_Uncertain) {
  auto snode = std::make_unique<SNode>(/*depth=*/1, /*t=*/SNodeType::place);

  IRBuilder builder;
  auto alloca = builder.create_local_var(PrimitiveType::i32);
  // Create two load statements to confuse the value_diff_ptr_index() pass.
  // This is probably a test of implementation, though...
  auto load1 = builder.create_local_load(alloca);
  auto load2 = builder.create_local_load(alloca);
  auto *gptr1 = builder.create_global_ptr(snode.get(), /*indices=*/{load1});
  auto *gptr2 = builder.create_global_ptr(snode.get(), /*indices=*/{load2});

  const auto aa = alias_analysis(gptr1, gptr2);
  EXPECT_EQ(aa, AliasResult::uncertain);
}

TEST(AliasAnalysis, GlobalPtr_DiffSNodes) {
  auto snode1 = std::make_unique<SNode>(/*depth=*/1, /*t=*/SNodeType::place);
  auto snode2 = std::make_unique<SNode>(/*depth=*/1, /*t=*/SNodeType::place);

  IRBuilder builder;
  const auto indices = make_const_indices({1, 2, 3}, &builder);
  auto *gptr1 = builder.create_global_ptr(snode1.get(), indices);
  auto *gptr2 = builder.create_global_ptr(snode2.get(), indices);

  const auto aa = alias_analysis(gptr1, gptr2);
  EXPECT_EQ(aa, AliasResult::different);
}

// TODO(#2193): Add tests for other edge cases and other kinds of statements.

TEST(AliasAnalysis, ExternalPtr_Same) {
  IRBuilder builder;
  auto *arg1 = builder.create_arg_load(1, PrimitiveType::i32, true);
  auto *arg2 = builder.create_arg_load(2, PrimitiveType::i32, false);
  const auto indices = std::vector<Stmt *>{arg2, arg2};
  auto *eptr1 = builder.create_external_ptr(arg1, indices);
  auto *eptr2 = builder.create_external_ptr(arg1, indices);

  const auto aa = alias_analysis(eptr1, eptr2);
  EXPECT_EQ(aa, AliasResult::same);
}

TEST(AliasAnalysis, ExternalPtr_Different) {
  IRBuilder builder;
  auto *arg1 = builder.create_arg_load(1, PrimitiveType::i32, true);
  auto *arg2 = builder.create_arg_load(2, PrimitiveType::i32, false);
  const auto indices1 = std::vector<Stmt *>{arg2, builder.get_int32(1)};
  const auto indices2 = std::vector<Stmt *>{arg2, builder.get_int32(2)};
  auto *eptr1 = builder.create_external_ptr(arg1, indices1);
  auto *eptr2 = builder.create_external_ptr(arg1, indices2);

  const auto aa = alias_analysis(eptr1, eptr2);
  EXPECT_EQ(aa, AliasResult::different);
}

TEST(AliasAnalysis, ExternalPtr_Uncertain) {
  IRBuilder builder;
  auto *arg1 = builder.create_arg_load(1, PrimitiveType::i32, true);
  auto *arg2 = builder.create_arg_load(2, PrimitiveType::i32, false);
  auto *arg3 = builder.create_arg_load(3, PrimitiveType::i32, false);
  const auto indices1 = std::vector<Stmt *>{arg2, arg2};
  const auto indices2 = std::vector<Stmt *>{arg2, arg3};
  auto *eptr1 = builder.create_external_ptr(arg1, indices1);
  auto *eptr2 = builder.create_external_ptr(arg1, indices2);

  const auto aa = alias_analysis(eptr1, eptr2);
  EXPECT_EQ(aa, AliasResult::uncertain);
}

TEST(AliasAnalysis, ExternalPtr_DiffPtr) {
  IRBuilder builder;
  auto *arg1 = builder.create_arg_load(1, PrimitiveType::i32, true);
  auto *arg2 = builder.create_arg_load(2, PrimitiveType::i32, true);
  auto *arg3 = builder.create_arg_load(3, PrimitiveType::i32, false);
  const auto indices = std::vector<Stmt *>{arg3, arg3};
  auto *eptr1 = builder.create_external_ptr(arg1, indices);
  auto *eptr2 = builder.create_external_ptr(arg2, indices);

  const auto aa = alias_analysis(eptr1, eptr2);
  EXPECT_EQ(aa, AliasResult::uncertain);
}

}  // namespace analysis
}  // namespace irpass
}  // namespace lang
}  // namespace taichi
