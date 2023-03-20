#pragma once

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/Cloning.h"

#if defined(TI_WITH_AMDGPU)
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#endif

namespace taichi {
namespace lang {
using namespace llvm;

struct AddStructForFuncPass : public ModulePass {
  static inline char ID{0};
  std::string func_name_;
  int tls_size_;
  AddStructForFuncPass(std::string func_name, int tls_size) : ModulePass(ID) {
    func_name_ = func_name;
    tls_size_ = tls_size;
  }
  bool runOnModule(llvm::Module &M) override {
    auto struct_for_func = M.getFunction("parallel_struct_for");
    auto &llvm_context = M.getContext();
    auto value_map = llvm::ValueToValueMapTy();
    auto patched_struct_for_func =
        llvm::CloneFunction(struct_for_func, value_map);
    patched_struct_for_func->setName(func_name_);

    int num_found_alloca = 0;
    llvm::AllocaInst *alloca = nullptr;

    auto char_type = llvm::Type::getInt8Ty(llvm_context);

    // Find the "1" in "char tls_buffer[1]" and replace it with
    // "tls_buffer_size"
    for (auto &bb : *patched_struct_for_func) {
      for (llvm::Instruction &inst : bb) {
        auto now_alloca = llvm::dyn_cast<AllocaInst>(&inst);
        if (!now_alloca || now_alloca->getAlign().value() != 8)
          continue;
        auto alloca_type = now_alloca->getAllocatedType();
        // Allocated type should be array [1 x i8]
        if (alloca_type->isArrayTy() &&
            alloca_type->getArrayNumElements() == 1 &&
            alloca_type->getArrayElementType() == char_type) {
          alloca = now_alloca;
          num_found_alloca++;
        }
      }
    }
    TI_ASSERT(num_found_alloca == 1 && alloca);
    auto new_type = llvm::ArrayType::get(char_type, tls_size_);
    llvm::IRBuilder<> builder(alloca);
    auto *new_alloca = builder.CreateAlloca(new_type);
    new_alloca->setAlignment(Align(8));
    TI_ASSERT(alloca->hasOneUse());
    auto *gep = llvm::cast<llvm::GetElementPtrInst>(alloca->user_back());
    TI_ASSERT(gep->getPointerOperand() == alloca);
    std::vector<Value *> indices(gep->idx_begin(), gep->idx_end());
    builder.SetInsertPoint(gep);
    auto *new_gep = builder.CreateInBoundsGEP(new_type, new_alloca, indices);
    gep->replaceAllUsesWith(new_gep);
    gep->eraseFromParent();
    alloca->eraseFromParent();
    return false;
  }
};

#if defined(TI_WITH_AMDGPU)
struct AMDGPUConvertAllocaInstAddressSpacePass : public FunctionPass {
  static inline char ID{0};
  AMDGPUConvertAllocaInstAddressSpacePass() : FunctionPass(ID) {
  }
  bool runOnFunction(llvm::Function &f) override {
    f.addFnAttr("target-cpu",
                "gfx" + AMDGPUContext::get_instance().get_mcpu().substr(3, 4));
    f.addFnAttr("target-features", "");
    for (auto &bb : f) {
      std::vector<AllocaInst *> alloca_inst_vec;
      for (Instruction &inst : bb) {
        AllocaInst *now_alloca = dyn_cast<AllocaInst>(&inst);
        if (!now_alloca ||
            now_alloca->getType()->getAddressSpace() != (unsigned)0) {
          continue;
        }
        alloca_inst_vec.push_back(now_alloca);
      }
      for (auto &allocainst : alloca_inst_vec) {
        auto alloca_type = allocainst->getAllocatedType();
        llvm::IRBuilder<> builder(allocainst);
        // magic number 5 and 0 represent `private`,`generic` respectively
        // more details, please ref
        // https://llvm.org/docs/AMDGPUUsage.html#address-spaces
        auto *new_alloca = builder.CreateAlloca(alloca_type, (unsigned)5);
        auto new_type = llvm::PointerType::get(alloca_type, (unsigned)0);
        new_alloca->setAlignment(Align(allocainst->getAlign().value()));
        auto *addrspacecast = builder.CreateAddrSpaceCast(new_alloca, new_type);
        allocainst->replaceAllUsesWith(addrspacecast);
        allocainst->eraseFromParent();
      }
    }
    return false;
  }
};

struct AMDGPUAddStructForFuncPass : public ModulePass {
  static inline char ID{0};
  std::string func_name_;
  int tls_size_;
  AMDGPUAddStructForFuncPass(std::string func_name, int tls_size)
      : ModulePass(ID) {
    func_name_ = func_name;
    tls_size_ = tls_size;
  }
  bool runOnModule(llvm::Module &M) override {
    auto struct_for_func = M.getFunction("parallel_struct_for");
    auto &llvm_context = M.getContext();
    auto value_map = llvm::ValueToValueMapTy();
    auto patched_struct_for_func =
        llvm::CloneFunction(struct_for_func, value_map);
    patched_struct_for_func->setName(func_name_);

    int num_found_alloca = 0;
    llvm::AllocaInst *alloca = nullptr;

    auto char_type = llvm::Type::getInt8Ty(llvm_context);

    // Find the "1" in "char tls_buffer[1]" and replace it with
    // "tls_buffer_size"
    for (auto &bb : *patched_struct_for_func) {
      for (llvm::Instruction &inst : bb) {
        auto now_alloca = llvm::dyn_cast<AllocaInst>(&inst);
        if (!now_alloca || now_alloca->getAlign().value() != 8)
          continue;
        auto alloca_type = now_alloca->getAllocatedType();
        // Allocated type should be array [1 x i8]
        if (alloca_type->isArrayTy() &&
            alloca_type->getArrayNumElements() == 1 &&
            alloca_type->getArrayElementType() == char_type) {
          alloca = now_alloca;
          num_found_alloca++;
        }
      }
    }
    TI_ASSERT(num_found_alloca == 1 && alloca);
    auto new_type = llvm::ArrayType::get(char_type, tls_size_);
    llvm::IRBuilder<> builder(alloca);
    // Find the allocInst `alloca i32, tls_size, align 8` and replace it with
    //   alloca i32, tls_size, align 8, addrspace(5)
    //   addrspacecast i32 addrspace(5)* %n to i32*

    // ditto magic number 5 and 0
    // more details, please ref
    // https://llvm.org/docs/AMDGPUUsage.html#address-spaces
    auto *new_alloca = builder.CreateAlloca(new_type, (unsigned)5);
    new_alloca->setAlignment(Align(8));
    auto new_ty = llvm::PointerType::get(new_type, unsigned(0));
    auto *new_cast = builder.CreateAddrSpaceCast(new_alloca, new_ty);
    new_alloca->setAlignment(Align(8));
    TI_ASSERT(alloca->hasOneUse());
    auto *cast = llvm::cast<llvm::AddrSpaceCastInst>(alloca->user_back());
    TI_ASSERT(cast->hasOneUse());
    auto *gep = llvm::cast<llvm::GetElementPtrInst>(cast->user_back());
    TI_ASSERT(gep->getPointerOperand() == cast);
    std::vector<Value *> indices(gep->idx_begin(), gep->idx_end());
    builder.SetInsertPoint(gep);
    auto *new_gep = builder.CreateInBoundsGEP(new_type, new_cast, indices);
    gep->replaceAllUsesWith(new_gep);
    gep->eraseFromParent();
    cast->eraseFromParent();
    alloca->eraseFromParent();
    return false;
  }
};

struct AMDGPUConvertFunctionBodyAllocsAddressSpacePass : public FunctionPass {
  static inline char ID{0};
  AMDGPUConvertFunctionBodyAllocsAddressSpacePass() : FunctionPass(ID) {
  }
  bool runOnFunction(llvm::Function &f) override {
    for (auto &bb : f) {
      if (bb.getName() != "allocs")
        continue;

      std::vector<AllocaInst *> alloca_inst_vec;
      for (Instruction &inst : bb) {
        AllocaInst *now_alloca = dyn_cast<AllocaInst>(&inst);
        if (!now_alloca ||
            now_alloca->getType()->getAddressSpace() != (unsigned)0) {
          continue;
        }
        alloca_inst_vec.push_back(now_alloca);
      }
      for (auto &allocainst : alloca_inst_vec) {
        auto alloca_type = allocainst->getAllocatedType();
        llvm::IRBuilder<> builder(allocainst);
        auto *new_alloca = builder.CreateAlloca(alloca_type, (unsigned)5);
        auto new_type = llvm::PointerType::get(alloca_type, (unsigned)0);
        new_alloca->setAlignment(Align(allocainst->getAlign().value()));
        auto *addrspacecast = builder.CreateAddrSpaceCast(new_alloca, new_type);
        allocainst->replaceAllUsesWith(addrspacecast);
        allocainst->eraseFromParent();
      }
    }
    return false;
  }
};

struct AMDGPUConvertFuncParamAddressSpacePass : public ModulePass {
  static inline char ID{0};
  AMDGPUConvertFuncParamAddressSpacePass() : ModulePass(ID) {
  }
  bool runOnModule(llvm::Module &M) override {
    for (auto &f : M) {
      bool is_kernel = false;
      const std::string func_name = f.getName().str();
      if (starts_with(func_name, "runtime_")) {
        f.setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
        // ref https://llvm.org/docs/AMDGPUUsage.html
        // “amdgpu-flat-work-group-size”=”min,max”
        // Specify the minimum and maximum flat work group sizes that will be
        // specified when the kernel is dispatched. Generated by the
        // amdgpu_flat_work_group_size CLANG attribute [CLANG-ATTR]. The implied
        // default value is 1,1024.
        f.addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
        is_kernel = true;
      }
      if (!is_kernel && !f.isDeclaration())
        f.setLinkage(llvm::Function::PrivateLinkage);
    }
    std::vector<llvm::Function *> kernel_function;
    for (auto &f : M) {
      if (f.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL)
        kernel_function.push_back(&f);
    }
    for (auto &f : kernel_function) {
      llvm::FunctionType *func_type = f->getFunctionType();
      std::vector<llvm::Type *> new_func_params;
      for (auto &arg : f->args()) {
        if (arg.getType()->getTypeID() == llvm::Type::PointerTyID) {
          // This is a temporary LLVM interface to handle transition from typed
          // pointer to opaque pointer In the future, if we only clang++ > 14,
          // we can compeletely comply to opaque pointer and replace the
          // following code with llvm::PointerType::get(M.getContext(),
          // usigned(1))
          auto new_type = llvm::PointerType::getWithSamePointeeType(
              llvm::dyn_cast<llvm::PointerType>(arg.getType()), unsigned(1));

          new_func_params.push_back(new_type);
        } else {
          new_func_params.push_back(arg.getType());
        }
      }
      auto new_func_type = llvm::FunctionType::get(func_type->getReturnType(),
                                                   new_func_params, false);
      auto new_func = llvm::Function::Create(new_func_type, f->getLinkage(),
                                             f->getAddressSpace());
      new_func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      new_func->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
      new_func->addFnAttr(
          "target-cpu",
          "gfx" + AMDGPUContext::get_instance().get_mcpu().substr(3, 4));
      new_func->setComdat(f->getComdat());
      f->getParent()->getFunctionList().insert(f->getIterator(), new_func);
      new_func->takeName(f);
      new_func->getBasicBlockList().splice(new_func->begin(),
                                           f->getBasicBlockList());
      for (llvm::Function::arg_iterator I = f->arg_begin(), E = f->arg_end(),
                                        I2 = new_func->arg_begin();
           I != E; ++I, ++I2) {
        if (I->getType()->getTypeID() == llvm::Type::PointerTyID) {
          auto &front_bb = new_func->getBasicBlockList().front();
          llvm::Instruction *addrspacecast =
              new AddrSpaceCastInst(I2, I->getType());
          front_bb.getInstList().insertAfter(front_bb.getFirstInsertionPt(),
                                             addrspacecast);
          I->replaceAllUsesWith(addrspacecast);
          I2->takeName(&*I);
        } else {
          I->replaceAllUsesWith(&*I2);
          I2->takeName(&*I);
        }
      }

      f->eraseFromParent();
    }
    return false;
  }
};

#endif

}  // namespace lang
}  // namespace taichi
