// A LLVM JIT compiler wrapper
#pragma once

// Based on
// https://llvm.org/docs/tutorial/BuildingAJIT3.html

// Note that
// https://llvm.org/docs/tutorial/BuildingAJIT2.html
// leads to a JIT that crashes all C++ exception after JIT session
// destruction...

#if defined(min)
#undef min
#endif
#if defined(max)
#undef max
#endif
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
#include <memory>
#include "../tlang_util.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;
using namespace llvm::orc;

std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module);
int compile_ptx_and_launch(const std::string &ptx,
                           const std::string &kernel_name,
                           void *);
void global_optimize_module_cpu(std::unique_ptr<llvm::Module> &module);

class TaichiLLVMJITCPU {
 private:
  ExecutionSession ES;
  std::map<VModuleKey, std::shared_ptr<SymbolResolver>> Resolvers;
  std::unique_ptr<TargetMachine> TM;
  const DataLayout DL;
  LegacyRTDyldObjectLinkingLayer ObjectLayer;
  LegacyIRCompileLayer<decltype(ObjectLayer), SimpleCompiler> CompileLayer;

  using OptimizeFunction = std::function<std::unique_ptr<llvm::Module>(
      std::unique_ptr<llvm::Module>)>;

  LegacyIRTransformLayer<decltype(CompileLayer), OptimizeFunction>
      OptimizeLayer;

  std::unique_ptr<JITCompileCallbackManager> CompileCallbackManager;
  LegacyCompileOnDemandLayer<decltype(OptimizeLayer)> CODLayer;

 public:
  TaichiLLVMJITCPU(JITTargetMachineBuilder JTMB, DataLayout DL)
      : TM(EngineBuilder().selectTarget()),
        DL(TM->createDataLayout()),
        ObjectLayer(ES,
                    [this](VModuleKey K) {
                      return LegacyRTDyldObjectLinkingLayer::Resources{
                          std::make_shared<SectionMemoryManager>(),
                          Resolvers[K]};
                    }),
        CompileLayer(ObjectLayer, SimpleCompiler(*TM)),
        OptimizeLayer(CompileLayer,
                      [this](std::unique_ptr<llvm::Module> M) {
                        return optimizeModule(std::move(M));
                      }),
        CompileCallbackManager(cantFail(
            orc::createLocalCompileCallbackManager(TM->getTargetTriple(),
                                                   ES,
                                                   0))),
        CODLayer(ES,
                 OptimizeLayer,
                 [&](orc::VModuleKey K) { return Resolvers[K]; },
                 [&](orc::VModuleKey K, std::shared_ptr<SymbolResolver> R) {
                   Resolvers[K] = std::move(R);
                 },
                 [](Function &F) { return std::set<Function *>({&F}); },
                 *CompileCallbackManager,
                 orc::createLocalIndirectStubsManagerBuilder(
                     TM->getTargetTriple())) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  const DataLayout &getDataLayout() const {
    return DL;
  }

  static Expected<std::unique_ptr<TaichiLLVMJITCPU>> create(Arch arch) {
    std::unique_ptr<JITTargetMachineBuilder> jtmb;
    if (arch == Arch::x64) {
      auto JTMB = JITTargetMachineBuilder::detectHost();
      if (!JTMB)
        return JTMB.takeError();
      jtmb = std::make_unique<JITTargetMachineBuilder>(std::move(*JTMB));
    } else {
      TI_ASSERT(arch == Arch::cuda);
      Triple triple("nvptx64", "nvidia", "cuda");
      jtmb = std::make_unique<JITTargetMachineBuilder>(triple);
    }

    auto DL = jtmb->getDefaultDataLayoutForTarget();
    if (!DL)
      return DL.takeError();

    return llvm::make_unique<TaichiLLVMJITCPU>(std::move(*jtmb),
                                               std::move(*DL));
  }

  VModuleKey addModule(std::unique_ptr<llvm::Module> M) {
    global_optimize_module_cpu(M);
    // Create a new VModuleKey.
    VModuleKey K = ES.allocateVModule();

    // Build a resolver and associate it with the new key.
    Resolvers[K] = createLegacyLookupResolver(
        ES,
        [this](const std::string &Name) -> JITSymbol {
          if (auto Sym = CompileLayer.findSymbol(Name, false))
            return Sym;
          else if (auto Err = Sym.takeError())
            return std::move(Err);
          if (auto SymAddr =
                  RTDyldMemoryManager::getSymbolAddressInProcess(Name))
            return JITSymbol(SymAddr, JITSymbolFlags::Exported);
          return nullptr;
        },
        [](Error Err) { cantFail(std::move(Err), "lookupFlags failed"); });

    // Add the module to the JIT with the new key.
    cantFail(CODLayer.addModule(K, std::move(M)));
    return K;
  }

  JITSymbol lookup(const std::string Name) {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return CODLayer.findSymbol(MangledNameStream.str(), true);
  }

  void removeModule(VModuleKey K) {
    cantFail(CODLayer.removeModule(K));
  }

 private:
  std::unique_ptr<llvm::Module> optimizeModule(
      std::unique_ptr<llvm::Module> M) {
    // Create a function pass manager.
    auto FPM = llvm::make_unique<legacy::FunctionPassManager>(M.get());

    // Add some optimizations.
    FPM->add(createInstructionCombiningPass());
    FPM->add(createReassociatePass());
    FPM->add(createGVNPass());
    FPM->add(createCFGSimplificationPass());
    FPM->doInitialization();

    // Run the optimizations over all functions in the module being added to
    // the JIT.
    for (auto &F : *M)
      FPM->run(F);

    return M;
  }

 public:
  std::size_t get_type_size(llvm::Type *type) {
    return DL.getTypeAllocSize(type);
  }
};

TLANG_NAMESPACE_END
