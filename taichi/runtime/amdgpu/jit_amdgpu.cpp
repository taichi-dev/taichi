#include "taichi/runtime/amdgpu/jit_amdgpu.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/runtime/llvm/llvm_context_pass.h"

namespace taichi {
namespace lang {

#if defined(TI_WITH_AMDGPU)
JITModule *JITSessionAMDGPU ::add_module(std::unique_ptr<llvm::Module> M,
                                         int max_reg) {
  auto hsaco = compile_module_to_hsaco(M);
  TI_TRACE("hsaco size: {:.2f}KB", hsaco.size() / 1024.0);

  void *amdgpu_module;
  auto t = Time::get_time();
  AMDGPUDriver::get_instance().module_load_data(&amdgpu_module, hsaco.c_str());
  TI_TRACE("AMDGPU load data from module time : {}ms",
           (Time::get_time() - t) * 1000);
  modules.push_back(std::make_unique<JITModuleAMDGPU>(amdgpu_module));
  return modules.back().get();
}

std::string JITSessionAMDGPU::compile_module_to_hsaco(
    std::unique_ptr<llvm::Module> &llvm_module) {

  llvm::legacy::FunctionPassManager function_pass_manager_addrcast(llvm_module.get());
  function_pass_manager_addrcast.add(new AMDGPUConvertFunctionBodyAllocsAddressSpacePass());
  for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func)
    if (func->getName() == "function_body")
      function_pass_manager_addrcast.run(*func);

  if (llvm::verifyModule(*llvm_module, &llvm::errs())) {
    llvm_module->print(llvm::errs(), nullptr);
    TI_WARN("Module broken");
  }
  using namespace llvm;

  if (this->config_.print_kernel_llvm_ir) {
    static FileSequenceWriter writer("taichi_kernel_amdgpu_llvm_ir_{:04d}.ll",
                                     "unoptimized LLVM IR (AMDGPU)");
    writer.write(llvm_module.get());
  }
  auto triple_str = llvm_module->getTargetTriple();
  std::string error_str;
  auto target = llvm::TargetRegistry::lookupTarget(triple_str, error_str);
  llvm::TargetOptions options;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      triple_str, AMDGPUContext::get_instance().get_mcpu(), "", options,
      llvm::Reloc::PIC_, llvm::CodeModel::Small, llvm::CodeGenOpt::Aggressive));

  llvm_module->setDataLayout(machine->createDataLayout());

  llvm::legacy::FunctionPassManager function_pass_manager(llvm_module.get());
  llvm::legacy::PassManager module_pass_manager;

  module_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(
      machine->getTargetIRAnalysis()));
  function_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(
      machine->getTargetIRAnalysis()));

  llvm::PassManagerBuilder builder;
  builder.OptLevel = 3;
  builder.Inliner =
      llvm::createFunctionInliningPass(builder.OptLevel, 0, false);
  machine->adjustPassManager(builder);
  builder.populateFunctionPassManager(function_pass_manager);
  builder.populateModulePassManager(module_pass_manager);

  machine->Options.MCOptions.AsmVerbose = true;

  auto tmp_dir = get_tmp_dir();
  uint64 random_num = get_random_num();

  auto obj_filename = "taichi_amdgcn_" + std::to_string(random_num) + ".o";
  auto hsaco_filename =
      "taichi_amdgcn_" + std::to_string(random_num) + ".hsaco";
  auto obj_path = tmp_dir + obj_filename;
  auto hsaco_path = tmp_dir + hsaco_filename;
  std::error_code ec;

  llvm::SmallString<0> outstr;
  llvm::raw_svector_ostream llvm_stream(outstr);

  machine->addPassesToEmitFile(module_pass_manager, llvm_stream, nullptr,
                               llvm::CGFT_ObjectFile, true);
  function_pass_manager.doInitialization();
  for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func)
    function_pass_manager.run(*func);
  function_pass_manager.doFinalization();
  module_pass_manager.run(*llvm_module);

  std::string obj_str(outstr.begin(), outstr.end());
  std::ofstream(obj_path) << obj_str;

  TI_TRACE("Loading module...");
  [[maybe_unused]] auto _ = AMDGPUContext::get_instance().get_lock_guard();

  std::string lld_cmd = "ld.lld -shared " + obj_path + " -o " + hsaco_path;
  if (std::system(lld_cmd.c_str()))
    TI_ERROR(fmt::format("Generate {} Error", hsaco_filename));

  std::string hsaco_str = load_hsaco(hsaco_path);

  if (this->config_.print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer(
        "taichi_kernel_amdgpu_llvm_ir_optimized_{:04d}.ll",
        "unoptimized LLVM IR (AMDGPU)");
    writer.write(llvm_module.get());
  }
  return hsaco_str;
}

std::unique_ptr<JITSession> create_llvm_jit_session_amdgpu(
    TaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  TI_ASSERT(arch == Arch::amdgpu);
  auto data_layout = llvm::DataLayout(
      "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-"
      "v16:16-v24:32-"
      "v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-"
      "n32:64-S32-A5-G1-ni:7");
  return std::make_unique<JITSessionAMDGPU>(tlctx, config, data_layout);
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_amdgpu(
    TaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  TI_NOT_IMPLEMENTED
}
#endif

}  // namespace lang
}  // namespace taichi
