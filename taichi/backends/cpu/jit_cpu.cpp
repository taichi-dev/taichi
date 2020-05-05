// A LLVM JIT compiler for CPU archs wrapper

#include <memory>

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"

#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/jit/jit_session.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;
using namespace llvm::orc;

class JITSessionCPU;

// Upgrade to LLVM 10: https://github.com/taichi-dev/taichi/issues/655
#if LLVM_VERSION_MAJOR >= 10
class JITModuleCPU : public JITModule {
 private:
  JITSessionCPU *session;
  JITDylib *dylib;

 public:
  JITModuleCPU(JITSessionCPU *session, JITDylib *dylib)
      : session(session), dylib(dylib) {
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
  RTDyldObjectLinkingLayer object_layer;
  IRCompileLayer compile_layer;
  DataLayout DL;
  MangleAndInterner Mangle;
  std::mutex mut;
  std::vector<llvm::orc::JITDylib *> all_libs;
  int module_counter;
  SectionMemoryManager *memory_manager;

 public:
  JITSessionCPU(JITTargetMachineBuilder JTMB, DataLayout DL)
      : object_layer(ES,
                     [&]() {
                       auto smgr = std::make_unique<SectionMemoryManager>();
                       memory_manager = smgr.get();
                       return smgr;
                     }),
        compile_layer(ES, object_layer, ConcurrentIRCompiler(std::move(JTMB))),
        DL(DL),
        Mangle(ES, this->DL),
        module_counter(0),
        memory_manager(nullptr) {
  }

  ~JITSessionCPU() {
    std::lock_guard<std::mutex> _(mut);
    if (memory_manager)
      memory_manager->deregisterEHFrames();
  }

  DataLayout get_data_layout() override {
    return DL;
  }

  JITModule *add_module(std::unique_ptr<llvm::Module> M) override {
    TI_ASSERT(M);
    global_optimize_module_cpu(M);
    std::lock_guard<std::mutex> _(mut);
    auto &dylib = ES.createJITDylib(fmt::format("{}", module_counter));
    dylib.setGenerator(cantFail(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(DL)));
    auto *thread_safe_context = get_current_program()
                                    .get_llvm_context(host_arch())
                                    ->get_this_thread_thread_safe_context();
    cantFail(compile_layer.add(dylib, llvm::orc::ThreadSafeModule(
                                          std::move(M), *thread_safe_context)));
    all_libs.push_back(&dylib);
    auto new_module = std::make_unique<JITModuleCPU>(this, &dylib);
    auto new_module_raw_ptr = new_module.get();
    modules.push_back(std::move(new_module));
    module_counter++;
    return new_module_raw_ptr;
  }
  void *lookup(const std::string Name) override {
    std::lock_guard<std::mutex> _(mut);
    auto symbol = ES.lookup(all_libs, Mangle(Name));
    if (!symbol)
      TI_ERROR("Function \"{}\" not found", Name);
    return (void *)(symbol->getAddress());
  }

  void *lookup_in_module(JITDylib *lib, const std::string Name) {
    std::lock_guard<std::mutex> _(mut);
    auto symbol = ES.lookup({lib}, Mangle(Name));
    if (!symbol)
      TI_ERROR("Function \"{}\" not found", Name);
    return (void *)(symbol->getAddress());
  }

 private:
  static void global_optimize_module_cpu(std::unique_ptr<llvm::Module> &module);
};

void *JITModuleCPU::lookup_function(const std::string &name) {
  return session->lookup_in_module(dylib, name);
}
#else
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
  std::map<VModuleKey, std::shared_ptr<SymbolResolver> > resolvers;
  std::unique_ptr<TargetMachine> TM;
  DataLayout DL;
  LegacyRTDyldObjectLinkingLayer object_layer;
  LegacyIRCompileLayer<decltype(object_layer), SimpleCompiler> compile_layer;
  MangleAndInterner Mangle;
  std::mutex mut;

 public:
  JITSessionCPU(JITTargetMachineBuilder JTMB, DataLayout DL)
      : TM(EngineBuilder().selectTarget()),
        DL(DL),
        object_layer(ES,
                     [this](VModuleKey K) {
                       return LegacyRTDyldObjectLinkingLayer::Resources{
                           std::make_shared<SectionMemoryManager>(),
                           resolvers[K]};
                     }),
        compile_layer(object_layer, SimpleCompiler(*TM)),
        Mangle(ES, this->DL) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  DataLayout get_data_layout() override {
    return DL;
  }

  JITModule *add_module(std::unique_ptr<llvm::Module> M) override {
    TI_ASSERT(M);
    global_optimize_module_cpu(M);
    std::lock_guard<std::mutex> _(mut);
    // Create a new VModuleKey.
    VModuleKey K = ES.allocateVModule();

    // Build a resolver and associate it with the new key.
    resolvers[K] = createLegacyLookupResolver(
        ES,
        [this](const std::string &Name) -> JITSymbol {
          if (auto Sym = object_layer.findSymbol(Name, false))
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
    cantFail(compile_layer.addModule(K, std::move(M)));
    auto new_module = std::make_unique<JITModuleCPU>(this, K);
    auto new_module_raw_ptr = new_module.get();
    modules.push_back(std::move(new_module));
    return new_module_raw_ptr;
  }

  void *lookup(const std::string Name) override {
    std::lock_guard<std::mutex> _(mut);
    auto mangled = *Mangle(Name);
    auto symbol = object_layer.findSymbol(
        mangled, false);  // On Windows the last argument must be False
    if (!symbol)
      TI_ERROR("Function \"{}\" (mangled=\"{}\") not found", Name,
               std ::string(mangled));
    return (void *)(llvm::cantFail(symbol.getAddress()));
  }

  void *lookup_in_module(VModuleKey key, const std::string Name) {
    std::lock_guard<std::mutex> _(mut);
    auto mangled = *Mangle(Name);
    auto symbol = compile_layer.findSymbolIn(
        key, mangled, false);  // On Windows the last argument must be False
    if (!symbol)
      TI_ERROR("Function \"{}\" (mangled=\"{}\") not found", Name,
               std ::string(mangled));
    return (void *)(llvm::cantFail(symbol.getAddress()));
  }

 private:
  static void global_optimize_module_cpu(std::unique_ptr<llvm::Module> &module);
};

void *JITModuleCPU::lookup_function(const std::string &name) {
  return session->lookup_in_module(key, name);
}
#endif

void JITSessionCPU::global_optimize_module_cpu(
    std::unique_ptr<llvm::Module> &module) {
  TI_AUTO_PROF
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_ERROR("Module broken");
  }
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

  {
    TI_PROFILER("llvm_function_pass");
    function_pass_manager.doInitialization();
    for (llvm::Module::iterator i = module->begin(); i != module->end(); i++)
      function_pass_manager.run(*i);

    function_pass_manager.doFinalization();
  }

  {
    TI_PROFILER("llvm_module_pass");
    module_pass_manager.run(*module);
  }

  if (get_current_program().config.print_kernel_llvm_ir_optimized) {
    TI_INFO("Functions with > 100 instructions in optimized LLVM IR:");
    static int counter = 0;
    std::error_code ec;
    auto fn = fmt::format("taichi_optimized_{:04d}.ll", counter);
    llvm::raw_fd_ostream fdos(fn, ec);
    module->print(fdos, nullptr);
    TaichiLLVMContext::print_huge_functions(module.get());
    TI_INFO("Optimized LLVM IR emitted to file {}", fn);
    counter++;
  }
}

std::unique_ptr<JITSession> create_llvm_jit_session_cpu(Arch arch) {
  std::unique_ptr<JITTargetMachineBuilder> jtmb;
  TI_ASSERT(arch_is_cpu(arch));
  auto JTMB = JITTargetMachineBuilder::detectHost();
  if (!JTMB)
    TI_ERROR("LLVM TargetMachineBuilder has failed.");
  jtmb = std::make_unique<JITTargetMachineBuilder>(std::move(*JTMB));

  auto data_layout = jtmb->getDefaultDataLayoutForTarget();
  if (!data_layout) {
    TI_ERROR("LLVM TargetMachineBuilder has failed when getting data layout.");
  }

  return std::make_unique<JITSessionCPU>(std::move(*jtmb),
                                         std::move(*data_layout));
}

TLANG_NAMESPACE_END
