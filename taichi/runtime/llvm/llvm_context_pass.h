#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"

#if defined(TI_WITH_AMDGPU)
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#endif

namespace taichi {
namespace lang {
using namespace llvm;
struct AMDGPUConvertAllocaInstAddressSpacePass : public FunctionPass {
    static char ID;
    AMDGPUConvertAllocaInstAddressSpacePass() : FunctionPass(ID) {}
    bool runOnFunction(llvm::Function &f) override {
        f.addFnAttr("target-cpu", "gfx" + AMDGPUContext::get_instance().get_mcpu().substr(3,4));
        f.addFnAttr("target-features","");
        for (auto &bb: f) {
            std::vector<AllocaInst*> alloca_inst_vec;
            for (Instruction &inst : bb) {
                AllocaInst* now_alloca = dyn_cast<AllocaInst>(&inst);
                if (!now_alloca || 
                    now_alloca->getType()->getAddressSpace() != (unsigned)0) {
                continue;
                }
                alloca_inst_vec.push_back(now_alloca);
            }
            for (auto &allocainst : alloca_inst_vec) {
                auto alloca_type = allocainst->getAllocatedType();
                IRBuilder<> builder(allocainst);
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
    static char ID;
    AMDGPUConvertFuncParamAddressSpacePass() : ModulePass(ID) {}
    bool runOnModule(llvm::Module &M) override {
        for (auto &f : M) {
        bool is_kernel = false;
        const std::string func_name = f.getName().str();
        if (starts_with(func_name, "runtime_")) {
          f.setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
          f.addFnAttr("amdgpu-flat-work-group-size", "1, 256");
          is_kernel = true;
        }
        if (!is_kernel && !f.isDeclaration())
          f.setLinkage(llvm::Function::PrivateLinkage);
      }
      std::vector<llvm::Function *> global_func;
      for (auto &f : M) {
        if (f.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL)
          global_func.push_back(&f);
      }
      for (auto &f : global_func) {
        llvm::FunctionType *func_type = f->getFunctionType();
        std::vector<llvm::Type*> new_func_params;
        for (auto &arg : f->args()) {
          if (arg.getType()->getTypeID() == llvm::Type::PointerTyID) {
            auto new_type = llvm::PointerType::get(arg.getType()->getPointerElementType(), unsigned(1));
            new_func_params.push_back(new_type);
          }
          else {
            new_func_params.push_back(arg.getType());
          }
        }
        auto new_func_type = llvm::FunctionType::get(func_type->getReturnType(), new_func_params, false);
        auto new_func = llvm::Function::Create(new_func_type, f->getLinkage(), f->getAddressSpace());
        new_func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
        new_func->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
        new_func->addFnAttr("target-cpu", "gfx" + AMDGPUContext::get_instance().get_mcpu().substr(3,4));
        new_func->setComdat(f->getComdat());
        f->getParent()->getFunctionList().insert(f->getIterator(), new_func);
        new_func->takeName(f);
        new_func->getBasicBlockList().splice(new_func->begin(), f->getBasicBlockList());
        for (llvm::Function::arg_iterator I = f->arg_begin(), E = f->arg_end(),
                                    I2 = new_func->arg_begin(); I != E; ++I, ++I2) {
          if (I->getType()->getTypeID() == llvm::Type::PointerTyID) {
            auto &front_bb = new_func->getBasicBlockList().front();
            llvm::Instruction *addrspacecast = new AddrSpaceCastInst(I2, I->getType());
            front_bb.getInstList().insertAfter(front_bb.getFirstInsertionPt(), addrspacecast);
            I->replaceAllUsesWith(addrspacecast);
            I2->takeName(&*I);
          }
          else {
            I->replaceAllUsesWith(&*I2);
            I2->takeName(&*I);
          }
        }

        f->eraseFromParent();
      }
      return false;
    }
};

char AMDGPUConvertAllocaInstAddressSpacePass::ID = 0;
char AMDGPUConvertFuncParamAddressSpacePass::ID = 0;

} // namespace lang
} // namespace taichi