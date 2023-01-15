
#include "dx12_llvm_passes.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include "taichi/program/compile_config.h"
#include "taichi/runtime/llvm/llvm_context.h"

using namespace llvm;
using namespace taichi::lang::directx12;

#define DEBUG_TYPE "dxil-taichi-intrinsic-lower"

namespace {

class TaichiIntrinsicLower : public ModulePass {
 public:
  bool runOnModule(Module &M) override {
    auto &Ctx = M.getContext();
    // patch intrinsic
    auto patch_intrinsic = [&](std::string name, Intrinsic::ID intrin,
                               bool ret = true,
                               std::vector<llvm::Type *> types = {},
                               std::vector<llvm::Value *> extra_args = {}) {
      auto func = M.getFunction(name);
      if (!func) {
        return;
      }
      func->deleteBody();
      auto bb = llvm::BasicBlock::Create(Ctx, "entry", func);
      IRBuilder<> builder(Ctx);
      builder.SetInsertPoint(bb);
      std::vector<llvm::Value *> args;
      for (auto &arg : func->args())
        args.push_back(&arg);
      args.insert(args.end(), extra_args.begin(), extra_args.end());
      if (ret) {
        builder.CreateRet(builder.CreateIntrinsic(intrin, types, args));
      } else {
        builder.CreateIntrinsic(intrin, types, args);
        builder.CreateRetVoid();
      }
      func->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
      taichi::lang::TaichiLLVMContext::mark_inline(func);
    };

    llvm::IRBuilder<> B(Ctx);
    Value *i32Zero = B.getInt32(0);

    auto patch_intrinsic_to_const = [&](std::string name, Constant *C,
                                        Type *Ty) {
      auto func = M.getFunction(name);
      if (!func) {
        return;
      }
      func->deleteBody();
      auto bb = llvm::BasicBlock::Create(Ctx, "entry", func);
      IRBuilder<> B(Ctx);
      B.SetInsertPoint(bb);
      Value *V = C;
      if (V->getType()->isPointerTy())
        V = B.CreateLoad(Ty, C);
      B.CreateRet(V);
      func->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
      taichi::lang::TaichiLLVMContext::mark_inline(func);
    };
    // group thread id.
    patch_intrinsic("thread_idx", Intrinsic::dx_thread_id_in_group, true, {},
                    {i32Zero});
    // group idx.
    patch_intrinsic("block_idx", Intrinsic::dx_group_id, true, {}, {i32Zero});
    // Group Size
    unsigned group_size = 64;
    if (config)
      group_size = config->default_gpu_block_dim;

    auto *I32Ty = B.getInt32Ty();
    Constant *block_dim = B.getInt32(group_size);
    patch_intrinsic_to_const("block_dim", block_dim, I32Ty);
    // Num work groups will be in a special CBuffer.
    // TaichiRuntimeContextLower pass will place the CBuffer to special binding
    // space.
    Type *TyNumWorkGroups = FixedVectorType::get(I32Ty, 3);
    Constant *CBNumWorkGroups = createGlobalVariableForResource(
        M, NumWorkGroupsCBName, TyNumWorkGroups);

    Constant *NumWorkGroupX = cast<Constant>(
        B.CreateConstGEP2_32(TyNumWorkGroups, CBNumWorkGroups, 0, 0));
    patch_intrinsic_to_const("grid_dim", NumWorkGroupX, I32Ty);
    return true;
  }

  TaichiIntrinsicLower(const taichi::lang::CompileConfig *config = nullptr)
      : ModulePass(ID), config(config) {
    initializeTaichiIntrinsicLowerPass(*PassRegistry::getPassRegistry());
  }

  static char ID;  // Pass identification.
 private:
  const taichi::lang::CompileConfig *config;
};
char TaichiIntrinsicLower::ID = 0;

}  // end anonymous namespace

INITIALIZE_PASS(TaichiIntrinsicLower,
                DEBUG_TYPE,
                "Lower taichi intrinsic",
                false,
                false)

llvm::ModulePass *llvm::createTaichiIntrinsicLowerPass(
    const taichi::lang::CompileConfig *config) {
  return new TaichiIntrinsicLower(config);
}
