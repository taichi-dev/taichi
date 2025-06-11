#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/SeparateConstOffsetFromGEP.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Target/TargetMachine.h"

// This is the crucial include for CUDA driver types like CUjit_option
#include <cuda.h>
#include <string>
#include <vector>

namespace taichi::lang {

#if defined(TI_WITH_CUDA)

// Helper function to check if runtime_initialize exists
bool module_has_runtime_initialize(
    const llvm::Module::FunctionListType &function_list) {
  for (const auto &func : function_list) {
    if (func.getName() == "runtime_initialize") {
      return true;
    }
  }
  return false;
}

// Helper function to get a representative name for dumping files
std::string moduleToDumpName(llvm::Module *M) {
  std::string dumpName = M->getName().str();
  if (M->getFunctionList().empty()) {
    return dumpName;
  }
  if (!module_has_runtime_initialize(M->getFunctionList())) {
    dumpName = M->getFunctionList().begin()->getName().str();
  }
  return dumpName;
}

JITModule *JITSessionCUDA::add_module(std::unique_ptr<llvm::Module> M,
                                      int max_reg) {
  const char *dump_ir_env = std::getenv("TAICHI_DUMP_IR");
  if (dump_ir_env) {
    const std::string dumpOutDir = "/tmp/ir/";
    std::filesystem::create_directories(dumpOutDir);
    std::string dumpName = moduleToDumpName(M.get());
    std::string filename = dumpOutDir + "/" + dumpName + "_before_ptx.ll";
    std::error_code EC;
    llvm::raw_fd_ostream dest_file(filename, EC);
    if (!EC) {
      M->print(dest_file, nullptr);
    } else {
      TI_WARN("Failed to dump LLVM IR to file {}: {}", filename, EC.message());
    }
  }

  auto ptx = compile_module_to_ptx(M);
  if (this->config_.print_kernel_asm) {
    static FileSequenceWriter writer("taichi_kernel_nvptx_{:04d}.ptx",
                                     "module NVPTX");
    writer.write(ptx);
  }

  if (dump_ir_env) {
    const std::string dumpOutDir = "/tmp/ptx/";
    std::filesystem::create_directories(dumpOutDir);
    std::string dumpName = moduleToDumpName(M.get());
    std::string filename = dumpOutDir + "/" + dumpName + ".ptx";
    std::ofstream out_file(filename);
    if (out_file.is_open()) {
      out_file << ptx;
      out_file.close();
      TI_INFO("PTX dumped to: {}", filename);
    }
  }

  const char *load_ptx_env = std::getenv("TAICHI_LOAD_PTX");
  if (load_ptx_env) {
    const std::string dumpOutDir = "/tmp/ptx/";
    std::string dumpName = moduleToDumpName(M.get());
    std::string filename = dumpOutDir + "/" + dumpName + ".ptx";
    std::ifstream in_file(filename);
    if (in_file.is_open()) {
      TI_INFO("Loading PTX from file: {}", filename);
      std::stringstream ptx_stream;
      ptx_stream << in_file.rdbuf();
      ptx = ptx_stream.str();
      in_file.close();
    } else {
      TI_WARN("Failed to open PTX file for loading: {}", filename);
    }
  }
  if (ptx.empty() || ptx.back() != '\0') {
    ptx += '\0';
  }

  CUDAContext::get_instance().make_current();
  void *cuda_module;
  TI_TRACE("PTX size: {:.2f}KB", ptx.size() / 1024.0);
  auto t = Time::get_time();
  TI_TRACE("Loading module...");
  [[maybe_unused]] auto _ = CUDAContext::get_instance().get_lock_guard();

  std::vector<CUjit_option> options;
  std::vector<void *> option_values;
  unsigned int max_reg_uint = max_reg;

  if (max_reg > 0) {
    options.push_back(CU_JIT_MAX_REGISTERS);
    option_values.push_back(reinterpret_cast<void *>(&max_reg_uint));
  }

  CUDADriver::get_instance().module_load_data_ex(&cuda_module, ptx.c_str(),
                                                 options.size(),
                                                 options.data(),
                                                 option_values.data());
  TI_TRACE("CUDA module load time : {}ms", (Time::get_time() - t) * 1000);

  modules.push_back(std::make_unique<JITModuleCUDA>(cuda_module));
  return modules.back().get();
}

std::string cuda_mattrs() {
  return "+ptx" +
         std::to_string(CUDAContext::get_instance().get_mcpu_version());
}

std::string convert_name_for_ptx(std::string new_name) {
  for (char &i : new_name) {
    if (i == '@' || i == '?' || i == '$' || i == '<' || i == '>' ||
        (!std::isalnum(i) && i != '_' && i != '.')) {
      i = '_';
    }
  }
  if (!new_name.empty() && !isalpha(new_name[0]) && new_name[0] != '_' &&
      new_name[0] != '.') {
    new_name = "_" + new_name;
  }
  return new_name;
}

std::string JITSessionCUDA::compile_module_to_ptx(
    std::unique_ptr<llvm::Module> &module) {
  TI_AUTO_PROF
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

  for (auto &f : module->globals()) {
    f.setName(convert_name_for_ptx(f.getName().str()));
  }
  for (auto &f : *module) {
    f.setName(convert_name_for_ptx(f.getName().str()));
  }

  llvm::Triple triple(module->getTargetTriple());
  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  TargetOptions options;
  if (this->config_.fast_math) {
    options.AllowFPOpFusion = FPOpFusion::Fast;
    options.UnsafeFPMath = true;
    options.NoInfsFPMath = true;
    options.NoNaNsFPMath = true;
  }
  options.HonorSignDependentRoundingFPMathOption = false;
  options.NoZerosInBSS = false;
  options.GuaranteedTailCallOpt = false;

  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), CUDAContext::get_instance().get_mcpu(), cuda_mattrs(),
      options, llvm::Reloc::PIC_, llvm::CodeModel::Small,
      config_.opt_level > 0 ? llvm::CodeGenOptLevel::Default
                           : llvm::CodeGenOptLevel::None));

  TI_ERROR_UNLESS(target_machine, "Could not allocate target machine!");
  module->setDataLayout(target_machine->createDataLayout());

  const auto kFTZDenorms = 1;
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        kFTZDenorms);
  if (kFTZDenorms) {
    for (llvm::Function &fn : *module) {
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
      fn.addFnAttr("unsafe-fp-math", "true");
    }
  }

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB(target_machine.get());

  FAM.registerPass([&] { return target_machine->getTargetIRAnalysis(); });

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  OptimizationLevel opt_level = OptimizationLevel::O3;
  ModulePassManager MPM;
  if (config_.opt_level > 0) {
    MPM = PB.buildPerModuleDefaultPipeline(opt_level);
  }

  FunctionPassManager FPM;
  FPM.addPass(createFunctionToLoopPassAdaptor(LoopStrengthReducePass()));
  FPM.addPass(createFunctionToLoopPassAdaptor(IndVarSimplifyPass()));
  FPM.addPass(SeparateConstOffsetFromGEPPass(false));
  FPM.addPass(EarlyCSEPass(true));
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  SmallString<0> outstr;
  raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  target_machine->Options.MCOptions.AsmVerbose = true;

  if (target_machine->addPassesToEmitFile(MPM, ostream, nullptr,
                                         CodeGenFileType::AssemblyFile, true)) {
    TI_ERROR("Failed to set up passes to emit PTX source\n");
  }

  {
    TI_PROFILER("llvm_module_pass");
    MPM.run(*module, MAM);
  }

  if (this->config_.print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer(
        "taichi_kernel_cuda_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  return std::string(outstr.str());
}

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    TaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  TI_ASSERT(arch == Arch::cuda);
  auto data_layout = TaichiLLVMContext::get_data_layout(arch);
  return std::make_unique<JITSessionCUDA>(tlctx, config, data_layout);
}
#endif

}  // namespace taichi::lang