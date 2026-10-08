#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {

TEST(CompilerBookkeeping, ASTLoweringKeepsExistingUsers) {
  auto ir = std::make_unique<Block>();
  auto old = ir->push_back<FrontendAllocaStmt>(Identifier(0, "x"), PrimitiveType::i32);
  auto load = ir->push_back<LocalLoadStmt>(old);
  auto result = ir->push_back<ReturnStmt>(std::vector<Stmt *>{load});
  irpass::lower_ast(ir.get());
  ASSERT_TRUE(ir->statements[0]->is<AllocaStmt>());
  EXPECT_EQ(load->as<LocalLoadStmt>()->src, ir->statements[0].get());
  EXPECT_EQ(result->as<ReturnStmt>()->values[0], load);
}

TEST(CompilerBookkeeping, CSEHoistingAndSubsequentReplacements) {
  IRBuilder builder;
  auto condition = builder.create_arg_load({0}, PrimitiveType::i32, false, 0);
  auto input = builder.create_arg_load({1}, PrimitiveType::i32, false, 0);
  auto dest = builder.create_local_var(PrimitiveType::i32);
  auto branch = builder.create_if(condition);
  for (bool side : {true, false}) {
    auto guard = builder.get_if_guard(branch, side);
    auto seven = builder.get_int32(7);
    auto a = builder.create_add(input, seven);
    auto b = builder.create_add(input, seven);
    builder.create_local_store(dest, builder.create_add(a, b));
  }
  auto ir = builder.extract_ir();
  irpass::type_check(ir.get(), CompileConfig());
  EXPECT_TRUE(irpass::whole_kernel_cse(ir.get()));
  irpass::analysis::verify(ir.get());
  int additions = 0, stores = 0;
  for (auto &stmt : ir->statements) {
    additions += stmt->is<BinaryOpStmt>();
    stores += stmt->is<LocalStoreStmt>();
  }
  EXPECT_EQ(additions, 2);
  EXPECT_EQ(stores, 1);
}

}  // namespace taichi::lang
