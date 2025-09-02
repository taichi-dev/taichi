
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/llvm/llvm_context.h"

#include "llvm/Passes/PassBuilder.h"

namespace taichi::lang {

#if defined(TI_WITH_CUDA)

JITModule *JITSessionCUDA ::add_module(std::unique_ptr<llvm::Module> M,
                                       int max_reg) {
  auto ptx = compile_module_to_ptx(M);
  if (this->config_.print_kernel_asm) {
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
  [[maybe_unused]] auto _ = CUDAContext::get_instance().get_lock_guard();

  constexpr int max_num_options = 8;
  int num_options = 0;
  uint32 options[max_num_options];
  void *option_values[max_num_options];

  // Insert options
  if (max_reg != 0) {
    options[num_options] = CU_JIT_MAX_REGISTERS;
    option_values[num_options] = &max_reg;
    num_options++;
  }

  TI_ASSERT(num_options <= max_num_options);

  CUDADriver::get_instance().module_load_data_ex(
      &cuda_module, ptx.c_str(), num_options, options, option_values);
  TI_TRACE("CUDA module load time : {}ms", (Time::get_time() - t) * 1000);

  modules.push_back(std::make_unique<JITModuleCUDA>(cuda_module));
  return modules.back().get();
}

std::string cuda_mattrs() {
  // TODO: upgrade to ptx78 as supported by LLVM 16
  return "+ptx63";
}

std::string convert(std::string new_name) {
  // Evil C++ mangling on Windows will lead to "unsupported characters in
  // symbol" error in LLVM PTX printer. Convert here.
  for (int i = 0; i < (int)new_name.size(); i++) {
    if (new_name[i] == '@') {
      new_name.replace(i, 1, "_at_");
    } else if (new_name[i] == '?') {
      new_name.replace(i, 1, "_qm_");
    } else if (new_name[i] == '$') {
      new_name.replace(i, 1, "_dl_");
    } else if (new_name[i] == '<') {
      new_name.replace(i, 1, "_lb_");
    } else if (new_name[i] == '>') {
      new_name.replace(i, 1, "_rb_");
    } else if (!std::isalpha(new_name[i]) && !std::isdigit(new_name[i]) &&
               new_name[i] != '_' && new_name[i] != '.') {
      new_name.replace(i, 1, "_xx_");
    }
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
    TI_ERROR("LLVM Module broken");
  }

  using namespace llvm;

  if (this->config_.print_kernel_llvm_ir) {
    static FileSequenceWriter writer("taichi_kernel_cuda_llvm_ir_{:04d}.ll",
                                     "unoptimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  for (auto &f : module->globals())
    f.setName(convert(f.getName().str()));
  for (auto &f : *module)
    f.setName(convert(f.getName().str()));

  llvm::Triple triple(module->getTargetTriple());

  // Allocate target machine
  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  TargetOptions options;
  if (this->config_.fast_math) {
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

#if LLVM_VERSION_MAJOR >= 18
  const auto opt_level = llvm::CodeGenOptLevel::Aggressive;
#else
  const auto opt_level = llvm::CodeGenOpt::Aggressive;
#endif
  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), CUDAContext::get_instance().get_mcpu(), cuda_mattrs(),
      options, llvm::Reloc::PIC_, llvm::CodeModel::Small, opt_level));

  TI_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

  module->setTargetTriple(triple.str());
  module->setDataLayout(target_machine->createDataLayout());

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
      /* nvptx-f32ftz was deprecated.
       *
       * https://github.com/llvm/llvm-project/commit/a4451d88ee456304c26d552749aea6a7f5154bde#diff-6fda74ef428299644e9f49a2b0994c0d850a760b89828f655030a114060d075a
       */
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");

      // Use unsafe fp math for sqrt.approx instead of sqrt.rn
      fn.addFnAttr("unsafe-fp-math", "true");
    }
  }

  // Create the new analysis manager
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  // Create the new pass builder
  llvm::PipelineTuningOptions PTO;
  PTO.LoopInterleaving = false;
  PTO.LoopVectorization = false;
  PTO.SLPVectorization = true;
  PTO.LoopUnrolling = false;
  PTO.ForgetAllSCEVInLoopUnroll = true;

  llvm::PassBuilder PB(target_machine.get(), PTO);

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  target_machine->registerPassBuilderCallbacks(PB);

  llvm::ModulePassManager MPM =
      PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);

  {
    TI_PROFILER("llvm_module_pass");
    MPM.run(*module, MAM);
  }

  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_ERROR("LLVM Module broken");
  }

  if (this->config_.print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer(
        "taichi_kernel_cuda_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  llvm::SmallString<8> outstr;
  raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  llvm::legacy::PassManager LPM;
  LPM.add(createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));

  // Override default to generate verbose assembly.
  target_machine->Options.MCOptions.AsmVerbose = true;

#if LLVM_VERSION_MAJOR >= 18
  const auto file_type = llvm::CodeGenFileType::AssemblyFile;
#else
  const auto file_type = llvm::CGFT_AssemblyFile;
#endif
  bool fail = target_machine->addPassesToEmitFile(LPM, ostream, nullptr,
                                                  file_type, true);

  TI_ERROR_IF(fail, "Failed to set up passes to emit PTX source\n");
  LPM.run(*module);

  std::string buffer(outstr.begin(), outstr.end());
  buffer.push_back(0);
  return buffer;
}

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    TaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  TI_ASSERT(arch == Arch::cuda);
  // https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#data-layout
  auto data_layout = TaichiLLVMContext::get_data_layout(arch);
  return std::make_unique<JITSessionCUDA>(tlctx, config, data_layout);
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    TaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  TI_NOT_IMPLEMENTED
}
#endif

}  // namespace taichi::lang
