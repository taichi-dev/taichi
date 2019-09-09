#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm_jit.h"

TLANG_NAMESPACE_BEGIN

// From Halide:CodeGen_PTX_Dev.cpp

std::string mcpu() {
  return "sm_61";
  /*
  if (target.has_feature(Target::CUDACapability61)) {
    return "sm_61";
  } else if (target.has_feature(Target::CUDACapability50)) {
    return "sm_50";
  } else if (target.has_feature(Target::CUDACapability35)) {
    return "sm_35";
  } else if (target.has_feature(Target::CUDACapability32)) {
    return "sm_32";
  } else if (target.has_feature(Target::CUDACapability30)) {
    return "sm_30";
  } else {
    return "sm_20";
  }
  */
}

std::string mattrs() {
  return "+ptx50";
  /*
  if (target.has_feature(Target::CUDACapability61)) {
  } else if (target.features_any_of({Target::CUDACapability32,
                                     Target::CUDACapability50})) {
    // Need ptx isa 4.0.
    return "+ptx40";
  } else {
    // Use the default. For llvm 3.5 it's ptx 3.2.
    return "";
  }
  */
}

std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module) {
  using namespace llvm;

  // DISABLED - hooked in here to force PrintBeforeAll option - seems to be the
  // only way?
  /*char* argv[] = { "llc", "-print-before-all" };*/
  /*int argc = sizeof(argv)/sizeof(char*);*/
  /*cl::ParseCommandLineOptions(argc, argv, "Halide PTX internal compiler\n");*/

  llvm::Triple triple(module->getTargetTriple());

  // Allocate target machine

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

  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), mcpu(), mattrs(), options, llvm::Reloc::PIC_,
      llvm::CodeModel::Small, CodeGenOpt::Aggressive));

  TC_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

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
#define kDefaultDenorms 0
#define kFTZDenorms 1

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
  b.LoopVectorize = true;
  b.SLPVectorize = true;

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

  TC_ERROR_IF(fail, "Failed to set up passes to emit PTX source\n");

  // Run optimization passes
  function_pass_manager.doInitialization();
  for (llvm::Module::iterator i = module->begin(); i != module->end(); i++) {
    function_pass_manager.run(*i);
  }
  function_pass_manager.doFinalization();
  module_pass_manager.run(*module);

  TC_DEBUG("Done with CodeGen_PTX_Dev::compile_to_src");
  TC_DEBUG("PTX kernel: {}", outstr.c_str());

  std::string buffer(outstr.begin(), outstr.end());

  // Null-terminate the ptx source
  buffer.push_back(0);
  return buffer;
}

TLANG_NAMESPACE_END