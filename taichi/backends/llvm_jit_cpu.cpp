#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm_jit_cpu.h"
#include "../program.h"

TLANG_NAMESPACE_BEGIN

void global_optimize_module_cpu(std::unique_ptr<llvm::Module> &module) {
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

  auto t = Time::get_time();
  module_pass_manager.run(*module);
  t = Time::get_time() - t;
  // TI_INFO("Global optimization time: {} ms", t * 1000);
  if (get_current_program().config.print_kernel_llvm_ir_optimized) {
    TI_INFO("Global optimized IR:");
    module->print(llvm::errs(), nullptr);
  }
}


TLANG_NAMESPACE_END
