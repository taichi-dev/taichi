// A LLVM JIT compiler for CPU archs wrapper

#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
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
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "jit_session.h"

// Based on
// https://llvm.org/docs/tutorial/BuildingAJIT3.html
// (Note that
// https://llvm.org/docs/tutorial/BuildingAJIT2.html
// leads to a JIT that crashes all C++ exception after JIT session
// destruction.)

TLANG_NAMESPACE_BEGIN

using namespace llvm;
using namespace llvm::orc;

class JITSessionCPU;

class JITModuleCPU : public JITModule {
 private:
  JITSessionCPU *session;
  VModuleKey key;

 public:
  JITModuleCPU(JITSessionCPU *session, VModuleKey key)
      : session(session), key(key) {
  }

  void *lookup_function(const std::string &name) override;

  uint64 fetch_result_u64() override {
    // TODO: move this to a new class e.g. "RuntimeEnvironment"
    return *(uint64 *)get_current_program().result_buffer;
  }

  bool direct_dispatch() const override {
    return true;
  }
};

class JITSessionCPU : public JITSession {
 private:
  ExecutionSession ES;
  std::map<VModuleKey, std::shared_ptr<SymbolResolver>> resolvers;
  std::unique_ptr<TargetMachine> TM;
  const DataLayout DL;
  LegacyRTDyldObjectLinkingLayer object_layer;
  LegacyIRCompileLayer<decltype(object_layer), SimpleCompiler> compile_layer;

  using OptimizeFunction = std::function<std::unique_ptr<llvm::Module>(
      std::unique_ptr<llvm::Module>)>;

  LegacyIRTransformLayer<decltype(compile_layer), OptimizeFunction>
      OptimizeLayer;

  std::unique_ptr<JITCompileCallbackManager> CompileCallbackManager;
  LegacyCompileOnDemandLayer<decltype(OptimizeLayer)> CODLayer;

 public:
  JITSessionCPU(JITTargetMachineBuilder JTMB, DataLayout DL)
      : TM(EngineBuilder().selectTarget()),
        DL(TM->createDataLayout()),
        object_layer(ES,
                     [this](VModuleKey K) {
                       return LegacyRTDyldObjectLinkingLayer::Resources{
                           std::make_shared<SectionMemoryManager>(),
                           resolvers[K]};
                     }),
        compile_layer(object_layer, SimpleCompiler(*TM)),
        OptimizeLayer(compile_layer,
                      [this](std::unique_ptr<llvm::Module> M) {
                        return optimize_module(std::move(M));
                      }),
        CompileCallbackManager(cantFail(
            orc::createLocalCompileCallbackManager(TM->getTargetTriple(),
                                                   ES,
                                                   0))),
        CODLayer(ES,
                 OptimizeLayer,
                 [&](orc::VModuleKey K) { return resolvers[K]; },
                 [&](orc::VModuleKey K, std::shared_ptr<SymbolResolver> R) {
                   resolvers[K] = std::move(R);
                 },
                 [](Function &F) { return std::set<Function *>({&F}); },
                 *CompileCallbackManager,
                 orc::createLocalIndirectStubsManagerBuilder(
                     TM->getTargetTriple())) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  DataLayout get_data_layout() override {
    return DL;
  }

  JITModule *add_module(std::unique_ptr<llvm::Module> M) override {
    TI_ASSERT(M);
    global_optimize_module_cpu(M);
    // Create a new VModuleKey.
    VModuleKey K = ES.allocateVModule();

    // Build a resolver and associate it with the new key.
    resolvers[K] = createLegacyLookupResolver(
        ES,
        [this](const std::string &Name) -> JITSymbol {
          if (auto Sym = compile_layer.findSymbol(Name, false))
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
    auto new_module = std::make_unique<JITModuleCPU>(this, K);
    auto new_module_raw_ptr = new_module.get();
    modules.push_back(std::move(new_module));
    return new_module_raw_ptr;
  }

  void *lookup(const std::string Name) override {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    auto symbol = CODLayer.findSymbol(MangledNameStream.str(), true);
    if (!symbol)
      TI_ERROR("Function \"{}\" not found", Name);
    return (void *)(llvm::cantFail(symbol.getAddress()));
  }

  void *lookup_in_module(VModuleKey key, const std::string Name) {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    auto symbol = CODLayer.findSymbolIn(key, MangledNameStream.str(), true);
    if (!symbol)
      TI_ERROR("Function \"{}\" not found", Name);
    return (void *)(llvm::cantFail(symbol.getAddress()));
  }

  /*
  void remove_module(VModuleKey K) override {
    cantFail(CODLayer.removeModule(K));
  }
  */

 private:
  std::unique_ptr<llvm::Module> optimize_module(
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

  static void global_optimize_module_cpu(
      std::unique_ptr<llvm::Module> &module) {
    TI_AUTO_PROF
    auto JTMB = JITTargetMachineBuilder::detectHost();
    if (!JTMB) {
      TI_ERROR("Target machine creation failed.");
    }
    module->setTargetTriple(JTMB->getTargetTriple().str());
    llvm::Triple triple(module->getTargetTriple());

    std::string err_str;
    const llvm::Target *target =
        TargetRegistry::lookupTarget(triple.str(), err_str);
    TI_ERROR_UNLESS(target, err_str);

    TargetOptions options;
    options.PrintMachineCode = false;
    bool fast_math = get_current_program().config.fast_math;
    if (fast_math) {
      options.AllowFPOpFusion = FPOpFusion::Fast;
      options.UnsafeFPMath = 1;
      options.NoInfsFPMath = 1;
      options.NoNaNsFPMath = 1;
    } else {
      options.AllowFPOpFusion = FPOpFusion::Strict;
      options.UnsafeFPMath = 0;
      options.NoInfsFPMath = 0;
      options.NoNaNsFPMath = 0;
    }
    options.HonorSignDependentRoundingFPMathOption = false;
    options.NoZerosInBSS = false;
    options.GuaranteedTailCallOpt = false;
    options.StackAlignmentOverride = 0;

    legacy::FunctionPassManager function_pass_manager(module.get());
    legacy::PassManager module_pass_manager;

    llvm::StringRef mcpu = llvm::sys::getHostCPUName();
    std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
        triple.str(), mcpu.str(), "", options, llvm::Reloc::PIC_,
        llvm::CodeModel::Small, CodeGenOpt::Aggressive));

    TI_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

    module->setDataLayout(target_machine->createDataLayout());

    module_pass_manager.add(createTargetTransformInfoWrapperPass(
        target_machine->getTargetIRAnalysis()));
    function_pass_manager.add(createTargetTransformInfoWrapperPass(
        target_machine->getTargetIRAnalysis()));

    PassManagerBuilder b;
    b.OptLevel = 3;
    b.Inliner = createFunctionInliningPass(b.OptLevel, 0, false);
    b.LoopVectorize = true;
    b.SLPVectorize = true;

    target_machine->adjustPassManager(b);

    b.populateFunctionPassManager(function_pass_manager);
    b.populateModulePassManager(module_pass_manager);

    function_pass_manager.doInitialization();
    for (llvm::Module::iterator i = module->begin(); i != module->end(); i++)
      function_pass_manager.run(*i);

    function_pass_manager.doFinalization();

    {
      TI_PROFILER("llvm_module_pass");
      module_pass_manager.run(*module);
    }

    if (get_current_program().config.print_kernel_llvm_ir_optimized) {
      TI_INFO("Global optimized IR:");
      module->print(llvm::errs(), nullptr);
    }
  }
};

void *JITModuleCPU::lookup_function(const std::string &name) {
  return session->lookup_in_module(key, name);
}

std::unique_ptr<JITSession> create_llvm_jit_session_cpu(Arch arch) {
  std::unique_ptr<JITTargetMachineBuilder> jtmb;
  TI_ASSERT(arch_is_cpu(arch));
  auto JTMB = JITTargetMachineBuilder::detectHost();
  if (!JTMB)
    TI_ERROR("LLVM TargetMachineBuilder has failed.");
  jtmb = std::make_unique<JITTargetMachineBuilder>(std::move(*JTMB));

  auto DL = jtmb->getDefaultDataLayoutForTarget();
  if (!DL) {
    TI_ERROR("LLVM TargetMachineBuilder has failed when getting data layout.");
  }

  return llvm::make_unique<JITSessionCPU>(std::move(*jtmb), std::move(*DL));
}

TLANG_NAMESPACE_END
