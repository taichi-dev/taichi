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
#include "taichi/util/file_sequence_writer.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;
using namespace llvm::orc;

std::pair<JITTargetMachineBuilder, llvm::DataLayout> get_host_target_info() {
#if defined(TI_PLATFORM_OSX) and defined(TI_ARCH_ARM)
  // JITTargetMachineBuilder::detectHost() doesn't seem to work properly on
  // Apple M1 yet. Hence the hardcoded strings here.
  auto jtmb =
      JITTargetMachineBuilder(llvm::Triple("aarch64-apple-macosx11.0.0"));
  llvm::DataLayout data_layout("e-m:o-i64:64-i128:128-n32:64-S128");
#else
  auto expected_jtmb = JITTargetMachineBuilder::detectHost();
  if (!expected_jtmb)
    TI_ERROR("LLVM TargetMachineBuilder has failed.");
  auto jtmb = *expected_jtmb;
  auto expected_data_layout = jtmb.getDefaultDataLayoutForTarget();
  if (!expected_data_layout) {
    TI_ERROR("LLVM TargetMachineBuilder has failed when getting data layout.");
  }
  auto data_layout = *expected_data_layout;
#endif
  return std::make_pair(jtmb, data_layout);
}

class JITSessionCPU;

class JITModuleCPU : public JITModule {
 private:
  JITSessionCPU *session;
  JITDylib *dylib;

 public:
  JITModuleCPU(JITSessionCPU *session, JITDylib *dylib)
      : session(session), dylib(dylib) {
  }

  void *lookup_function(const std::string &name) override;

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
        compile_layer(ES,
                      object_layer,
                      std::make_unique<ConcurrentIRCompiler>(JTMB)),
        DL(DL),
        Mangle(ES, this->DL),
        module_counter(0),
        memory_manager(nullptr) {
    if (JTMB.getTargetTriple().isOSBinFormatCOFF()) {
      object_layer.setOverrideObjectFlagsWithResponsibilityFlags(true);
      object_layer.setAutoClaimResponsibilityForObjectSymbols(true);
    }
  }

  ~JITSessionCPU() {
    std::lock_guard<std::mutex> _(mut);
    if (memory_manager)
      memory_manager->deregisterEHFrames();
  }

  DataLayout get_data_layout() override {
    return DL;
  }

  void global_optimize_module(llvm::Module *module) override {
    global_optimize_module_cpu(module);
  }

  JITModule *add_module(std::unique_ptr<llvm::Module> M, int max_reg) override {
    TI_ASSERT(max_reg == 0);  // No need to specify max_reg on CPUs
    TI_ASSERT(M);
    global_optimize_module_cpu(M.get());
    std::lock_guard<std::mutex> _(mut);
    auto &dylib = ES.createJITDylib(fmt::format("{}", module_counter));
    dylib.addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            DL.getGlobalPrefix())));
    auto *thread_safe_context = get_current_program()
                                    .get_llvm_program_impl()
                                    ->get_llvm_context(host_arch())
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
#ifdef __APPLE__
    auto symbol = ES.lookup(all_libs, Mangle(Name));
#else
    auto symbol = ES.lookup(all_libs, ES.intern(Name));
#endif
    if (!symbol)
      TI_ERROR("Function \"{}\" not found", Name);
    return (void *)(symbol->getAddress());
  }

  void *lookup_in_module(JITDylib *lib, const std::string Name) {
    std::lock_guard<std::mutex> _(mut);
#ifdef __APPLE__
    auto symbol = ES.lookup({lib}, Mangle(Name));
#else
    auto symbol = ES.lookup({lib}, ES.intern(Name));
#endif
    if (!symbol)
      TI_ERROR("Function \"{}\" not found", Name);
    return (void *)(symbol->getAddress());
  }

 private:
  static void global_optimize_module_cpu(llvm::Module *module);
};

void *JITModuleCPU::lookup_function(const std::string &name) {
  return session->lookup_in_module(dylib, name);
}

void JITSessionCPU::global_optimize_module_cpu(llvm::Module *module) {
  TI_AUTO_PROF
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_ERROR("Module broken");
  }

  auto triple = get_host_target_info().first.getTargetTriple();

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

  legacy::FunctionPassManager function_pass_manager(module);
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
    if (false) {
      TI_INFO("Functions with > 100 instructions in optimized LLVM IR:");
      TaichiLLVMContext::print_huge_functions(module);
    }
    static FileSequenceWriter writer(
        "taichi_kernel_cpu_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (CPU)");
    writer.write(module);
  }
}

std::unique_ptr<JITSession> create_llvm_jit_session_cpu(Arch arch) {
  TI_ASSERT(arch_is_cpu(arch));
  auto target_info = get_host_target_info();
  return std::make_unique<JITSessionCPU>(target_info.first, target_info.second);
}

TLANG_NAMESPACE_END
