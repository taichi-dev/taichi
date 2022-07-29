// A LLVM JIT compiler for CPU archs wrapper

#include <memory>

#ifdef TI_WITH_LLVM
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
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"

#ifdef TI_LLVM_15
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#else
#include "llvm/Support/TargetRegistry.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#endif

#endif

#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/jit/jit_session.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/runtime/llvm/llvm_context.h"

TLANG_NAMESPACE_BEGIN

#ifdef TI_WITH_LLVM
using namespace llvm;
using namespace llvm::orc;
#endif

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
  JITSessionCPU *session_;
  JITDylib *dylib_;

 public:
  JITModuleCPU(JITSessionCPU *session, JITDylib *dylib)
      : session_(session), dylib_(dylib) {
  }

  void *lookup_function(const std::string &name) override;

  bool direct_dispatch() const override {
    return true;
  }
};

class JITSessionCPU : public JITSession {
 private:
  ExecutionSession es_;
  RTDyldObjectLinkingLayer object_layer_;
  IRCompileLayer compile_layer_;
  DataLayout dl_;
  MangleAndInterner mangle_;
  std::mutex mut_;
  std::vector<llvm::orc::JITDylib *> all_libs_;
  int module_counter_;
  SectionMemoryManager *memory_manager_;

 public:
#ifdef TI_LLVM_15
  JITSessionCPU(TaichiLLVMContext *tlctx,
                std::unique_ptr<ExecutorProcessControl> EPC,
                CompileConfig *config,
                JITTargetMachineBuilder JTMB,
                DataLayout DL)
      : JITSession(tlctx, config),
        es_(std::move(EPC)),
        object_layer_(es_,
                      [&]() {
                        auto smgr = std::make_unique<SectionMemoryManager>();
                        memory_manager_ = smgr.get();
                        return smgr;
                      }),
        compile_layer_(es_,
                       object_layer_,
                       std::make_unique<ConcurrentIRCompiler>(JTMB)),
        dl_(DL),
        mangle_(es_, this->dl_),
        module_counter_(0),
        memory_manager_(nullptr) {
    if (JTMB.getTargetTriple().isOSBinFormatCOFF()) {
      object_layer_.setOverrideObjectFlagsWithResponsibilityFlags(true);
      object_layer_.setAutoClaimResponsibilityForObjectSymbols(true);
    }
  }
#else
  JITSessionCPU(TaichiLLVMContext *tlctx,
                CompileConfig *config,
                JITTargetMachineBuilder JTMB,
                DataLayout DL)
      : JITSession(tlctx, config),
        object_layer_(es_,
                      [&]() {
                        auto smgr = std::make_unique<SectionMemoryManager>();
                        memory_manager_ = smgr.get();
                        return smgr;
                      }),
        compile_layer_(es_,
                       object_layer_,
                       std::make_unique<ConcurrentIRCompiler>(JTMB)),
        dl_(DL),
        mangle_(es_, this->dl_),
        module_counter_(0),
        memory_manager_(nullptr) {
    if (JTMB.getTargetTriple().isOSBinFormatCOFF()) {
      object_layer_.setOverrideObjectFlagsWithResponsibilityFlags(true);
      object_layer_.setAutoClaimResponsibilityForObjectSymbols(true);
    }
  }
#endif

  ~JITSessionCPU() override {
    std::lock_guard<std::mutex> _(mut_);
    if (memory_manager_)
      memory_manager_->deregisterEHFrames();
#ifdef TI_LLVM_15
    if (auto Err = es_.endSession())
      es_.reportError(std::move(Err));
#endif
  }

  DataLayout get_data_layout() override {
    return dl_;
  }

  void global_optimize_module(llvm::Module *module) override {
    global_optimize_module_cpu(module);
  }

  JITModule *add_module(std::unique_ptr<llvm::Module> M, int max_reg) override {
    TI_ASSERT(max_reg == 0);  // No need to specify max_reg on CPUs
    TI_ASSERT(M);
    global_optimize_module_cpu(M.get());
    std::lock_guard<std::mutex> _(mut_);
#ifdef TI_LLVM_15
    auto dylib_expect = es_.createJITDylib(fmt::format("{}", module_counter_));
    TI_ASSERT(dylib_expect);
    auto &dylib = dylib_expect.get();
#else
    auto &dylib = es_.createJITDylib(fmt::format("{}", module_counter_));
#endif
    dylib.addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            dl_.getGlobalPrefix())));
    auto *thread_safe_context =
        this->tlctx_->get_this_thread_thread_safe_context();
    cantFail(compile_layer_.add(
        dylib,
        llvm::orc::ThreadSafeModule(std::move(M), *thread_safe_context)));
    all_libs_.push_back(&dylib);
    auto new_module = std::make_unique<JITModuleCPU>(this, &dylib);
    auto new_module_raw_ptr = new_module.get();
    modules.push_back(std::move(new_module));
    module_counter_++;
    return new_module_raw_ptr;
  }

  void *lookup(const std::string Name) override {
    std::lock_guard<std::mutex> _(mut_);
#ifdef __APPLE__
    auto symbol = es_.lookup(all_libs_, mangle_(Name));
#else
    auto symbol = es_.lookup(all_libs_, es_.intern(Name));
#endif
    if (!symbol)
      TI_ERROR("Function \"{}\" not found", Name);
    return (void *)(symbol->getAddress());
  }

  void *lookup_in_module(JITDylib *lib, const std::string Name) {
    std::lock_guard<std::mutex> _(mut_);
#ifdef __APPLE__
    auto symbol = es_.lookup({lib}, mangle_(Name));
#else
    auto symbol = es_.lookup({lib}, es_.intern(Name));
#endif
    if (!symbol)
      TI_ERROR("Function \"{}\" not found", Name);
    return (void *)(symbol->getAddress());
  }

 private:
  void global_optimize_module_cpu(llvm::Module *module);
};

void *JITModuleCPU::lookup_function(const std::string &name) {
  return session_->lookup_in_module(dylib_, name);
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
#ifndef TI_LLVM_15
  // PrintMachineCode is removed in https://reviews.llvm.org/D83275.
  options.PrintMachineCode = false;
#endif
  if (this->config_->fast_math) {
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
#ifndef TI_LLVM_15
  // StackAlignmentOverride is removed in https://reviews.llvm.org/D103048.
  options.StackAlignmentOverride = 0;
#endif

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

  /*
    Optimization for llvm::GetElementPointer:
    https://github.com/taichi-dev/taichi/issues/5472 The three other passes
    "loop-reduce", "ind-vars", "cse" serves as preprocessing for
    "separate-const-offset-gep".

    Note there's an update for "separate-const-offset-gep" in llvm-12.
  */
  module_pass_manager.add(llvm::createLoopStrengthReducePass());
  module_pass_manager.add(llvm::createIndVarSimplifyPass());
  module_pass_manager.add(llvm::createSeparateConstOffsetFromGEPPass(false));
  module_pass_manager.add(llvm::createEarlyCSEPass(true));

  {
    TI_PROFILER("llvm_module_pass");
    module_pass_manager.run(*module);
  }

  if (this->config_->print_kernel_llvm_ir_optimized) {
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

std::unique_ptr<JITSession> create_llvm_jit_session_cpu(
    TaichiLLVMContext *tlctx,
    CompileConfig *config,
    Arch arch) {
  TI_ASSERT(arch_is_cpu(arch));
  auto target_info = get_host_target_info();
#ifdef TI_LLVM_15
  auto EPC = SelfExecutorProcessControl::Create();
  TI_ASSERT(EPC);
  return std::make_unique<JITSessionCPU>(tlctx, std::move(*EPC), config,
                                         target_info.first, target_info.second);
#else
  return std::make_unique<JITSessionCPU>(tlctx, config, target_info.first,
                                         target_info.second);
#endif
}

TLANG_NAMESPACE_END
