
#include <memory>
#if defined(TI_WITH_CUDA)
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif
#include "llvm/ADT/StringRef.h"
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
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <taichi/backends/cuda/cuda_utils.h>
#include <taichi/backends/cuda/cuda_context.h>
#include <taichi/program/program.h>
#include <taichi/runtime/llvm/context.h>
#include <taichi/system/timer.h>
#include "taichi/lang_util.h"
#include "jit_session.h"

TLANG_NAMESPACE_BEGIN

#if defined(TI_WITH_CUDA)
class JITModuleCUDA : public JITModule {
 private:
  CUmodule module;

 public:
  explicit JITModuleCUDA(CUmodule module) : module(module) {
  }

  void *lookup_function(const std::string &name) override {
    // auto _ = cuda_context->get_guard();
    cuda_context->make_current();
    CUfunction func;
    auto t = Time::get_time();
    check_cuda_error(cuModuleGetFunction(&func, module, name.c_str()));
    t = Time::get_time() - t;
    TI_TRACE("Kernel {} compilation time: {}ms", name, t * 1000);
    return (void *)func;
  }

  void call(const std::string &name,
            const std::vector<void *> &arg_pointers) override {
    launch(name, 1, 1, arg_pointers);
  }

  virtual void launch(const std::string &name,
                      std::size_t grid_dim,
                      std::size_t block_dim,
                      const std::vector<void *> &arg_pointers) override {
    auto func = lookup_function(name);
    cuda_context->launch(func, name, arg_pointers, grid_dim, block_dim);
  }

  uint64 fetch_result_u64() override {
    uint64 ret;
    check_cuda_error(cudaMemcpy(&ret, get_current_program().result_buffer,
                                sizeof(ret), cudaMemcpyDeviceToHost));
    return ret;
  }

  bool direct_dispatch() const override {
    return false;
  }
};

class JITSessionCUDA : public JITSession {
 public:
  llvm::DataLayout DL;

  explicit JITSessionCUDA(llvm::DataLayout data_layout) : DL(data_layout) {
  }

  virtual JITModule *add_module(std::unique_ptr<llvm::Module> M) override {
    auto ptx = compile_module_to_ptx(M);
    // auto _ = cuda_context->get_guard();
    cuda_context->make_current();
    // Create module for object
    CUmodule cudaModule;
    TI_TRACE("PTX size: {:.2f}KB", ptx.size() / 1024.0);
    auto t = Time::get_time();
    TI_TRACE("Loading module...");
    auto _ = std::lock_guard<std::mutex>(cuda_context->lock);
    check_cuda_error(
        cuModuleLoadDataEx(&cudaModule, ptx.c_str(), 0, nullptr, nullptr));
    TI_TRACE("CUDA module load time : {}ms", (Time::get_time() - t) * 1000);
    // cudaModules.push_back(cudaModule);
    modules.push_back(std::make_unique<JITModuleCUDA>(cudaModule));
    return modules.back().get();
  }

  virtual llvm::DataLayout get_data_layout() override {
    return DL;
  }

  static std::string compile_module_to_ptx(
      std::unique_ptr<llvm::Module> &module);
};

std::string cuda_mattrs() {
  return "+ptx50";
}

std::string convert(std::string new_name) {
  // Evil C++ mangling on Windows will lead to "unsupported characters in
  // symbol" error in LLVM PTX printer. Convert here.
  for (int i = 0; i < (int)new_name.size(); i++) {
    if (new_name[i] == '@')
      new_name.replace(i, 1, "_at_");
    if (new_name[i] == '?')
      new_name.replace(i, 1, "_qm_");
    if (new_name[i] == '$')
      new_name.replace(i, 1, "_dl_");
    if (new_name[i] == '<')
      new_name.replace(i, 1, "_lb_");
    if (new_name[i] == '>')
      new_name.replace(i, 1, "_rb_");
    TI_ASSERT(std::isalpha(new_name[i]) || std::isdigit(new_name[i]) ||
              new_name[i] == '_' || new_name[i] == '.');
  }
  TI_ASSERT(isalpha(new_name[0]) || new_name[0] == '_' || new_name[0] == '.');
  return new_name;
}

std::string JITSessionCUDA::compile_module_to_ptx(
    std::unique_ptr<llvm::Module> &module) {
  // Part of this function is borrowed from Halide::CodeGen_PTX_Dev.cpp
  using namespace llvm;

  for (auto &f : module->globals())
    f.setName(convert(f.getName()));
  for (auto &f : *module)
    f.setName(convert(f.getName()));

  llvm::Triple triple(module->getTargetTriple());

  // Allocate target machine

  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  bool fast_math = get_current_program().config.fast_math;

  TargetOptions options;
  options.PrintMachineCode = 0;
  if (fast_math) {
    options.AllowFPOpFusion = FPOpFusion::Fast;
    // See NVPTXISelLowering.cpp
    // Setting UnsafeFPMath true will result in approximations such as
    // sqrt.approx in PTX for both f32 and f64
    options.UnsafeFPMath = 1;
    options.NoInfsFPMath = 1;
    options.NoNaNsFPMath = 1;
  } else {
    options.AllowFPOpFusion = FPOpFusion::Strict;
    options.UnsafeFPMath = 0;
    options.NoInfsFPMath = 0;
    options.NoNaNsFPMath = 0;
  }
  options.HonorSignDependentRoundingFPMathOption = 0;
  options.NoZerosInBSS = 0;
  options.GuaranteedTailCallOpt = 0;
  options.StackAlignmentOverride = 0;

  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), cuda_context->get_mcpu(), cuda_mattrs(), options,
      llvm::Reloc::PIC_, llvm::CodeModel::Small, CodeGenOpt::Aggressive));

  TI_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

  module->setDataLayout(target_machine->createDataLayout());

  // Set up passes
  llvm::SmallString<8> outstr;
  raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  legacy::FunctionPassManager function_pass_manager(module.get());
  legacy::PassManager module_pass_manager;

  module_pass_manager.add(createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));
  function_pass_manager.add(createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));

  // NVidia's libdevice library uses a __nvvm_reflect to choose
  // how to handle denormalized numbers. (The pass replaces calls
  // to __nvvm_reflect with a constant via a map lookup. The inliner
  // pass then resolves these situations to fast code, often a single
  // instruction per decision point.)
  //
  // The default is (more) IEEE like handling. FTZ mode flushes them
  // to zero. (This may only apply to single-precision.)
  //
  // The libdevice documentation covers other options for math accuracy
  // such as replacing division with multiply by the reciprocal and
  // use of fused-multiply-add, but they do not seem to be controlled
  // by this __nvvvm_reflect mechanism and may be flags to earlier compiler
  // passes.
  const auto kFTZDenorms = 1;

  // Insert a module flag for the FTZ handling.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        kFTZDenorms);

  if (kFTZDenorms) {
    for (llvm::Function &fn : *module) {
      fn.addFnAttr("nvptx-f32ftz", "true");
    }
  }

  PassManagerBuilder b;
  b.OptLevel = 3;
  b.Inliner = createFunctionInliningPass(b.OptLevel, 0, false);
  b.LoopVectorize = false;
  b.SLPVectorize = false;

  target_machine->adjustPassManager(b);

  b.populateFunctionPassManager(function_pass_manager);
  b.populateModulePassManager(module_pass_manager);

  // Override default to generate verbose assembly.
  target_machine->Options.MCOptions.AsmVerbose = true;

  // Output string stream

  // Ask the target to add backend passes as necessary.
  bool fail = target_machine->addPassesToEmitFile(
      module_pass_manager, ostream, nullptr, TargetMachine::CGFT_AssemblyFile,
      true);

  TI_ERROR_IF(fail, "Failed to set up passes to emit PTX source\n");

  // Run optimization passes
  function_pass_manager.doInitialization();
  for (llvm::Module::iterator i = module->begin(); i != module->end(); i++) {
    function_pass_manager.run(*i);
  }
  function_pass_manager.doFinalization();
  module_pass_manager.run(*module);

  std::string buffer(outstr.begin(), outstr.end());

  // Null-terminate the ptx source
  buffer.push_back(0);
  return buffer;
}

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(Arch arch) {
  TI_ASSERT(arch == Arch::cuda);
  // TODO: assuming CUDA has the same data layout as the host arch
  std::unique_ptr<llvm::orc::JITTargetMachineBuilder> jtmb;
  auto JTMB = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!JTMB)
    TI_ERROR("LLVM TargetMachineBuilder has failed.");
  jtmb = std::make_unique<llvm::orc::JITTargetMachineBuilder>(std::move(*JTMB));

  auto DL = jtmb->getDefaultDataLayoutForTarget();
  if (!DL) {
    TI_ERROR("LLVM TargetMachineBuilder has failed when getting data layout.");
  }
  return std::make_unique<JITSessionCUDA>(DL.get());
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_cuda(Arch arch) {
  TI_NOT_IMPLEMENTED
}
#endif

TLANG_NAMESPACE_END
