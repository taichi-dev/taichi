#include "gtest/gtest.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "taichi/codegen/llvm/llvm_codegen_utils.h"

TEST(IsSameTypeTest, IsSameType) {
  llvm::LLVMContext ctx;
  auto integer = llvm::Type::getInt32Ty(ctx);
  auto structtype = llvm::StructType::create({integer, integer}, "structtype");
  auto structtype1 =
      llvm::StructType::create({integer, integer}, "structtype.1");
  auto structtype2 =
      llvm::StructType::create({integer, integer}, "structtype.2");
  auto structtype11 =
      llvm::StructType::create({integer, integer}, "structtype.11");
  auto structtype12 =
      llvm::StructType::create({integer, integer}, "structtype.12");
  std::vector<llvm::Type *> types = {structtype, structtype1, structtype2,
                                     structtype11, structtype12};
  for (auto t1 : types) {
    for (auto t2 : types) {
      ASSERT_TRUE(taichi::lang::is_same_type(t1, t2));
      ASSERT_TRUE(taichi::lang::is_same_type(llvm::PointerType::get(t1, 0),
                                             llvm::PointerType::get(t2, 0)));
    }
  }
  auto func1 = llvm::FunctionType::get(
      structtype, {structtype1, structtype2, structtype11, structtype12},
      false);
  auto func2 = llvm::FunctionType::get(
      structtype12, {structtype11, structtype1, structtype2, structtype},
      false);
  ASSERT_TRUE(taichi::lang::is_same_type(func1, func2));
}
