// A LLVM JIT compiler wrapper

#pragma once

#if __cpluscplus >= 201703L
static_assert(false, "please use C++17.");
#endif

// https://llvm.org/docs/tutorial/BuildingAJIT2.html
#include "../util.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
#include <memory>

TLANG_NAMESPACE_BEGIN

using namespace llvm;
using namespace llvm::orc;

class TaichiLLVMJIT {
 private:
  ExecutionSession ES;
  RTDyldObjectLinkingLayer ObjectLayer;
  IRCompileLayer CompileLayer;
  IRTransformLayer OptimizeLayer;

  DataLayout DL;
  MangleAndInterner Mangle;
  ThreadSafeContext Ctx;

 public:
  TaichiLLVMJIT(JITTargetMachineBuilder JTMB, DataLayout DL)
      : ObjectLayer(ES,
                    []() { return llvm::make_unique<SectionMemoryManager>(); }),
        CompileLayer(ES, ObjectLayer, ConcurrentIRCompiler(std::move(JTMB))),
        OptimizeLayer(ES, CompileLayer, optimizeModule),
        DL(std::move(DL)),
        Mangle(ES, this->DL),
        Ctx(llvm::make_unique<LLVMContext>()) {
    ES.getMainJITDylib().setGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(DL)));
  }

  const DataLayout &getDataLayout() const {
    return DL;
  }

  LLVMContext &getContext() {
    return *Ctx.getContext();
  }

  static Expected<std::unique_ptr<TaichiLLVMJIT>> create(Arch arch) {
    std::unique_ptr<JITTargetMachineBuilder> jtmb;
    if (arch == Arch::x86_64) {
      auto JTMB = JITTargetMachineBuilder::detectHost();
      if (!JTMB)
        return JTMB.takeError();
      jtmb = std::make_unique<JITTargetMachineBuilder>(std::move(*JTMB));
    } else {
      TC_ASSERT(arch == Arch::gpu);
      Triple triple("nvptx64", "nvidia", "cuda");
      jtmb = std::make_unique<JITTargetMachineBuilder>(triple);
    }

    auto DL = jtmb->getDefaultDataLayoutForTarget();
    if (!DL)
      return DL.takeError();

    return llvm::make_unique<TaichiLLVMJIT>(std::move(*jtmb), std::move(*DL));
  }

  Error addModule(std::unique_ptr<Module> M) {
    return OptimizeLayer.add(ES.getMainJITDylib(),
                             ThreadSafeModule(std::move(M), Ctx));
  }

  Expected<JITEvaluatedSymbol> lookup(StringRef Name) {
    return ES.lookup({&ES.getMainJITDylib()}, Mangle(Name.str()));
  }

 private:
  static Expected<ThreadSafeModule> optimizeModule(
      ThreadSafeModule TSM,
      const MaterializationResponsibility &R) {
    // Create a function pass manager.
    auto FPM = llvm::make_unique<legacy::FunctionPassManager>(TSM.getModule());

    // Add some optimizations.
    // FPM->add(createFunctionInliningPass());
    FPM->add(createInstructionCombiningPass());
    FPM->add(createReassociatePass());
    FPM->add(createGVNPass());
    FPM->add(createCFGSimplificationPass());

    FPM->doInitialization();

    /*
    llvm::ModulePassManager MPM;
    llvm::ModuleAnalysisManager moduleAnalysisManager;
    MPM.addPass(createFunctionInliningPass());
    MPM.run(*TSM.getModule(), moduleAnalysisManager);
    */

    // Run the optimizations over all functions in the module being added to
    // the JIT.

    for (auto &F : *TSM.getModule()) {
      FPM->run(F);
      // TC_INFO("Function IR Optimized");
      // F.print(errs(), nullptr);
    }

    return TSM;
  }

 public:
  std::size_t get_type_size(llvm::Type *type) {
    return DL.getTypeAllocSize(type);
  }
};

inline void *jit_lookup_name(TaichiLLVMJIT *jit, const std::string &name) {
  llvm::ExitOnError exit_on_err;
  return (void *)exit_on_err(jit->lookup(name)).getAddress();
}

TLANG_NAMESPACE_END