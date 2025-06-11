
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/llvm/llvm_context.h"

namespace taichi::lang {

#if defined(TI_WITH_CUDA)

bool module_has_runtime_initialize(
    llvm::Module::FunctionListType &function_list) {
  for (auto &func : function_list) {
    if (func.getName() == "runtime_initialize") {
      return true;
    }
  }
  return false;
}

std::string moduleToDumpName(llvm::Module *M) {
  std::string dumpName(M->getName().begin(), M->getName().end());
  std::cout << "module get function list len:" << M->getFunctionList().size()
            << std::endl;
  auto func0 = M->getFunctionList().begin();
  std::cout << "function 0 name: " << func0->getName().str() << std::endl;
  if (!module_has_runtime_initialize(M->getFunctionList())) {
    dumpName = std::string(func0->getName().begin(), func0->getName().end());
  }
  return dumpName;
}

JITModule *JITSessionCUDA ::add_module(std::unique_ptr<llvm::Module> M,
                                       int max_reg) {
  const char *dump_ir_env = std::getenv("TAICHI_DUMP_IR");
  if (dump_ir_env != nullptr) {
    const std::string dumpOutDir = "/tmp/ir/";
    std::filesystem::create_directories(dumpOutDir);
    std::string dumpName = moduleToDumpName(M.get());
    std::string filename = dumpOutDir + "/" + dumpName + "_before_ptx.ll";
    std::error_code EC;
    llvm::raw_fd_ostream dest_file(filename, EC);
    if (!EC) {
      M->print(dest_file, nullptr);
    } else {
      std::cout << "problem dumping file " << filename << ": " << EC.message()
                << std::endl;
      TI_ERROR("Failed to dump LLVM IR to file: {}", filename);
    }
  }

  auto ptx = compile_module_to_ptx(M);
  if (this->config_.print_kernel_asm) {
    static FileSequenceWriter writer("taichi_kernel_nvptx_{:04d}.ptx",
                                     "module NVPTX");
    writer.write(ptx);
  }

  if (dump_ir_env != nullptr) {
    const std::string dumpOutDir = "/tmp/ptx/";
    std::filesystem::create_directories(dumpOutDir);
    std::string dumpName = moduleToDumpName(M.get());
    std::string filename = dumpOutDir + "/" + dumpName + ".ptx";
    std::ofstream out_file(filename);
    if (out_file.is_open()) {
      out_file << ptx << std::endl;
      out_file.close();
    }
    std::cout << "########################## PTX dumped to: " << filename
              << std::endl;
  }

  const char *load_ptx_env = std::getenv("TAICHI_LOAD_PTX");
  if (load_ptx_env != nullptr) {
    const std::string dumpOutDir = "/tmp/ptx/";
    std::string dumpName = moduleToDumpName(M.get());
    std::string filename = dumpOutDir + "/" + dumpName + ".ptx";
    std::ifstream in_file(filename);
    if (in_file.is_open()) {
      TI_INFO("########################## Loading PTX from file: {}", filename);
      std::ostringstream ptx_stream;
      std::string line;
      while (std::getline(in_file, line)) {
        ptx_stream.write(line.c_str(), line.size());
        ptx_stream.write("\n", 1);
      }
      ptx_stream.write("\0", 1);  // Null-terminate the stream
      ptx = ptx_stream.str();
      in_file.close();
    } else {
      TI_WARN("Failed to open PTX file for loading: {}", filename);
    }
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
  // cudaModules.push_back(cudaModule);
  modules.push_back(std::make_unique<JITModuleCUDA>(cuda_module));
  return modules.back().get();
}

std::string cuda_mattrs() {
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
      /* nvptx-f32ftz was deprecated.
       *
       * https://github.com/llvm/llvm-project/commit/a4451d88ee456304c26d552749aea6a7f5154bde#diff-6fda74ef428299644e9f49a2b0994c0d850a760b89828f655030a114060d075a
       */
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");

      // Use unsafe fp math for sqrt.approx instead of sqrt.rn
      fn.addFnAttr("unsafe-fp-math", "true");
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

  // Ask the target to add backend passes as necessary.
  bool fail = target_machine->addPassesToEmitFile(
      module_pass_manager, ostream, nullptr, llvm::CGFT_AssemblyFile, true);

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

  if (this->config_.print_kernel_llvm_ir_optimized) {
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
