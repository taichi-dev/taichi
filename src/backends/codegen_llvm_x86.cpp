#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "../tlang_util.h"
#include "codegen_x86.h"
#include "../program.h"
#include "../ir.h"

#include "codegen_llvm.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;
class CodeGenLLVMCPU : public CodeGenLLVM {
 public:
  CodeGenLLVMCPU(CodeGenBase *codegen_base, Kernel *kernel)
      : CodeGenLLVM(codegen_base, kernel) {
  }
};

FunctionType CPUCodeGen::codegen_llvm() {
  TC_PROFILER("cpu codegen");
  return CodeGenLLVMCPU(this, kernel).gen();
}

void global_optimize_module_x86_64(std::unique_ptr<llvm::Module> &module) {
  TI_AUTO_PROF
  auto JTMB = JITTargetMachineBuilder::detectHost();
  if (!JTMB) {
    TC_ERROR("Target machine creation failed.");
  }
  module->setTargetTriple(JTMB->getTargetTriple().str());
  llvm::Triple triple(module->getTargetTriple());

  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TC_ERROR_UNLESS(target, err_str);

  TargetOptions options;
  options.PrintMachineCode = false;
  options.AllowFPOpFusion = FPOpFusion::Fast;
  options.UnsafeFPMath = true;
  options.NoInfsFPMath = true;
  options.NoNaNsFPMath = true;
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

  TC_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

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
  TC_INFO("Global optimization time: {} ms", t * 1000);
  // module->print(llvm::errs(), nullptr);
}


TLANG_NAMESPACE_END
