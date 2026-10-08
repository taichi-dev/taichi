#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {
namespace {

void forward_stores(Block *ir) {
  irpass::type_check(ir, CompileConfig());
  auto cfg = irpass::analysis::build_cfg(ir);
  cfg->simplify_graph();
  cfg->store_to_load_forwarding(false, false);
}

TEST(StoreForwarding, ErasedStoresAndUpdatedOperands) {
  IRBuilder builder;
  auto a = builder.create_local_var(PrimitiveType::i32);
  auto b = builder.create_local_var(PrimitiveType::i32);
  auto zero = builder.get_int32(0);
  auto seven = builder.get_int32(7);
  builder.create_local_store(a, zero);  // Redundant with alloca initialization.
  builder.create_local_store(a, seven);
  builder.create_local_store(a, seven);  // Removed before indexing next stmt.
  auto first = builder.create_local_load(a);
  builder.create_local_store(b, first);  // Operand is rewritten by forwarding.
  auto second = builder.create_local_load(b);
  auto result = builder.create_return(second);
  auto ir = builder.extract_ir();
  forward_stores(ir.get());
  EXPECT_EQ(result->values[0], seven);
  int stores = 0;
  for (auto &stmt : ir->statements) {
    EXPECT_FALSE(stmt->is<LocalLoadStmt>());
    stores += stmt->is<LocalStoreStmt>();
  }
  EXPECT_EQ(stores, 2);
}

TEST(StoreForwarding, BranchMergeKeepsConflictingDefinitions) {
  for (bool identical : {false, true}) {
    IRBuilder builder;
    auto condition = builder.create_arg_load({0}, PrimitiveType::i32, false, 0);
    auto a = builder.create_local_var(PrimitiveType::i32);
    auto one = builder.get_int32(1);
    auto two = builder.get_int32(2);
    auto branch = builder.create_if(condition);
    {
      auto guard = builder.get_if_guard(branch, true);
      builder.create_local_store(a, one);
    }
    {
      auto guard = builder.get_if_guard(branch, false);
      builder.create_local_store(a, identical ? one : two);
    }
    auto load = builder.create_local_load(a);
    auto result = builder.create_return(load);
    auto ir = builder.extract_ir();
    forward_stores(ir.get());
    EXPECT_EQ(result->values[0], identical ? static_cast<Stmt *>(one) : load);
  }
}

TEST(StoreForwarding, TensorElementRetainsAliasHandling) {
  auto ir = std::make_unique<Block>();
  auto tensor = TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::i32);
  auto a = ir->push_back<AllocaStmt>(tensor);
  auto one = ir->push_back<ConstStmt>(TypedConstant(1));
  auto two = ir->push_back<ConstStmt>(TypedConstant(2));
  auto values = ir->push_back<MatrixInitStmt>(std::vector<Stmt *>{one, two});
  values->ret_type = tensor;
  ir->push_back<LocalStoreStmt>(a, values);
  auto element = ir->push_back<MatrixPtrStmt>(a, one);
  auto load = ir->push_back<LocalLoadStmt>(element);
  auto result = ir->push_back<ReturnStmt>(std::vector<Stmt *>{load});
  forward_stores(ir.get());
  EXPECT_EQ(result->as<ReturnStmt>()->values[0], two);
}

TEST(StoreForwarding, CrossNodeAliasScanIncludesBlockPrefix) {
  IRBuilder builder;
  auto condition = builder.create_arg_load({0}, PrimitiveType::i32, false, 0);
  auto tensor = TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::i32);
  auto a = builder.create_local_var(tensor);
  auto other = builder.create_local_var(PrimitiveType::i32);
  auto one = builder.get_int32(1);
  auto two = builder.get_int32(2);
  auto values = builder.create_matrix_init({one, two});
  values->ret_type = tensor;
  builder.create_local_store(a, values);
  auto branch = builder.create_if(condition);
  {
    auto guard = builder.get_if_guard(branch, true);
    builder.create_local_store(other, one);
  }
  auto ir = builder.extract_ir();
  auto element = ir->push_back<MatrixPtrStmt>(a, one);
  auto load = ir->push_back<LocalLoadStmt>(element);
  auto result = ir->push_back<ReturnStmt>(std::vector<Stmt *>{load});
  forward_stores(ir.get());
  // The existing conservative cross-node alias check sees the allocation in
  // the prefix of this Block, before the current CFG node starts.
  EXPECT_EQ(result->as<ReturnStmt>()->values[0], load);
}

TEST(StoreForwarding, UnknownIncomingValueDoesNotHideLocalStore) {
  for (bool stored : {false, true}) {
    auto ir = std::make_unique<Block>();
    auto pointer_type = TypeFactory::get_instance().get_pointer_type(PrimitiveType::i32);
    auto address = ir->push_back<GlobalTemporaryStmt>(0, pointer_type);
    auto seven = ir->push_back<ConstStmt>(TypedConstant(7));
    if (stored) ir->push_back<GlobalStoreStmt>(address, seven);
    auto load = ir->push_back<GlobalLoadStmt>(address);
    auto result = ir->push_back<ReturnStmt>(std::vector<Stmt *>{load});
    forward_stores(ir.get());
    EXPECT_EQ(result->as<ReturnStmt>()->values[0], stored ? seven : load);
  }
}

}  // namespace
}  // namespace taichi::lang
