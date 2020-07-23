#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/program/program.h"
#include "taichi/runtime/llvm/context.h"
#include "taichi/system/timer.h"
#include "taichi/lang_util.h"
#include "taichi/jit/jit_session.h"
#include "taichi/util/file_sequence_writer.h"

TLANG_NAMESPACE_BEGIN

#if defined(TI_WITH_CUDA)
class JITModuleCUDA : public JITModule {
 private:
  void *module;

 public:
  explicit JITModuleCUDA(void *module) : module(module) {
  }

  void *lookup_function(const std::string &name) override {
    // TODO: figure out why using the guard leads to wrong tests results
    // auto context_guard = CUDAContext::get_instance().get_guard();
    CUDAContext::get_instance().make_current();
    void *func;
    auto t = Time::get_time();
    auto err = CUDADriver::get_instance().module_get_function.call_with_warning(
        &func, module, name.c_str());
    if (err) {
      TI_ERROR("Cannot look up function {} ", name);
    }
    t = Time::get_time() - t;
    TI_TRACE("CUDA module_get_function {} costs {} ms", name, t * 1000);
    return func;
  }

  void call(const std::string &name,
            const std::vector<void *> &arg_pointers) override {
    launch(name, 1, 1, 0, arg_pointers);
  }

  virtual void launch(const std::string &name,
                      std::size_t grid_dim,
                      std::size_t block_dim,
                      std::size_t shared_mem_bytes,
                      const std::vector<void *> &arg_pointers) override {
    auto func = lookup_function(name);
    CUDAContext::get_instance().launch(func, name, arg_pointers, grid_dim,
                                       block_dim, shared_mem_bytes);
  }

  bool direct_dispatch() const override {
    return false;
  }
};

class JITSessionCUDA : public JITSession {
 public:
  llvm::DataLayout data_layout;

  explicit JITSessionCUDA(llvm::DataLayout data_layout)
      : data_layout(data_layout) {
  }

  virtual JITModule *add_module(std::unique_ptr<llvm::Module> M) override {
    auto ptx = compile_module_to_ptx(M);
    if (get_current_program().config.print_kernel_nvptx) {
      static FileSequenceWriter writer("taichi_kernel_nvptx_{:04d}.ptx",
                                       "module NVPTX");
      writer.write(ptx);
    }
    // TODO: figure out why using the guard leads to wrong tests results
    // auto context_guard = CUDAContext::get_instance().get_guard();
    CUDAContext::get_instance().make_current();
    // Create module for object
    void *cuda_module;
    TI_TRACE("PTX size: {:.2f}KB", ptx.size() / 1024.0);
    auto t = Time::get_time();
    TI_TRACE("Loading module...");
    [[maybe_unused]] auto &&_ =
        std::move(CUDAContext::get_instance().get_lock_guard());
    CUDADriver::get_instance().module_load_data_ex(&cuda_module, ptx.c_str(), 0,
                                                   nullptr, nullptr);
    TI_TRACE("CUDA module load time : {}ms", (Time::get_time() - t) * 1000);
    // cudaModules.push_back(cudaModule);
    modules.push_back(std::make_unique<JITModuleCUDA>(cuda_module));
    return modules.back().get();
  }

  virtual llvm::DataLayout get_data_layout() override {
    return data_layout;
  }

  static std::string compile_module_to_ptx(
      std::unique_ptr<llvm::Module> &module);
};

std::string cuda_mattrs() {
  return "+ptx63";
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
  if (!new_name.empty())
    TI_ASSERT(isalpha(new_name[0]) || new_name[0] == '_' || new_name[0] == '.');
  return new_name;
}

std::string JITSessionCUDA::compile_module_to_ptx(
    std::unique_ptr<llvm::Module> &module) {
  TI_AUTO_PROF
  // Part of this function is borrowed from Halide::CodeGen_PTX_Dev.cpp
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_WARN("Module broken");
  }

  using namespace llvm;

  if (get_current_program().config.print_kernel_llvm_ir) {
    static FileSequenceWriter writer("taichi_kernel_cuda_llvm_ir_{:04d}.ll",
                                     "unoptimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

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
      triple.str(), CUDAContext::get_instance().get_mcpu(), cuda_mattrs(),
      options, llvm::Reloc::PIC_, llvm::CodeModel::Small,
      CodeGenOpt::Aggressive));

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
#if LLVM_VERSION_MAJOR >= 10
  bool fail = target_machine->addPassesToEmitFile(
      module_pass_manager, ostream, nullptr, llvm::CGFT_AssemblyFile, true);
#else
  bool fail = target_machine->addPassesToEmitFile(
      module_pass_manager, ostream, nullptr, TargetMachine::CGFT_AssemblyFile,
      true);
#endif

  TI_ERROR_IF(fail, "Failed to set up passes to emit PTX source\n");

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
    static FileSequenceWriter writer(
        "taichi_kernel_cuda_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  std::string buffer(outstr.begin(), outstr.end());

  // Null-terminate the ptx source
  buffer.push_back(0);
  return buffer;
}

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(Arch arch) {
  TI_ASSERT(arch == Arch::cuda);
  // https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#data-layout
  auto data_layout = llvm::DataLayout(
      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-"
      "f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  return std::make_unique<JITSessionCUDA>(data_layout);
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_cuda(Arch arch) {
  TI_NOT_IMPLEMENTED
}
#endif

TLANG_NAMESPACE_END
