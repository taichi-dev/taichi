#include <llvm/Transforms/Utils/Cloning.h>
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils.h"

#include "taichi_llvm_context.h"
#include "backends/llvm_jit.h"

TLANG_NAMESPACE_BEGIN

static llvm::ExitOnError exit_on_err;

TaichiLLVMContext::TaichiLLVMContext() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  ctx = std::make_unique<llvm::LLVMContext>();
  jit = exit_on_err(TaichiLLVMJIT::Create());
}

TaichiLLVMContext::~TaichiLLVMContext() {
}

llvm::Type *TaichiLLVMContext::get_data_type(DataType dt) {
  if (dt == DataType::i32) {
    return llvm::Type::getInt32Ty(*ctx);
  } else if (dt == DataType::f32) {
    return llvm::Type::getFloatTy(*ctx);
  } else {
    TC_INFO(data_type_name(dt));
    TC_NOT_IMPLEMENTED
  }
  return nullptr;
}

std::unique_ptr<llvm::Module> TaichiLLVMContext::clone_struct_module() {
  TC_ASSERT(struct_module);
  return llvm::CloneModule(*struct_module);
}

void TaichiLLVMContext::set_struct_module(
    const std::unique_ptr<llvm::Module> &module) {
  TC_ASSERT(module);
  struct_module = llvm::CloneModule(*module);
}

template <typename T>
llvm::Value *TaichiLLVMContext::get_constant(T t) {
  if constexpr (std::is_same_v<T, float32> || std::is_same_v<T, float64>) {
    return llvm::ConstantFP::get(*ctx, llvm::APFloat(t));
  } else if (std::is_same_v<T, int32> || std::is_same_v<T, uint32>) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(32, t, true));
  } else if (std::is_same_v<T, int64> || std::is_same_v<T, uint64>) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(64, t, true));
  } else {
    TC_NOT_IMPLEMENTED
    return nullptr;
  }
}

template llvm::Value *TaichiLLVMContext::get_constant(float32 t);
template llvm::Value *TaichiLLVMContext::get_constant(int32 t);
template llvm::Value *TaichiLLVMContext::get_constant(uint32 t);
template llvm::Value *TaichiLLVMContext::get_constant(int64 t);

TLANG_NAMESPACE_END
