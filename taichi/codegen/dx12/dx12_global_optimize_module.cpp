
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/runtime/llvm/llvm_context.h"

#include "dx12_llvm_passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Function.h"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

namespace taichi::lang {
namespace directx12 {

const char *NumWorkGroupsCBName = "num_work_groups.cbuf";

const llvm::StringRef ShaderAttrKindStr = "hlsl.shader";

void mark_function_as_cs_entry(::llvm::Function *F) {
  F->addFnAttr(ShaderAttrKindStr, "compute");
}
bool is_cs_entry(::llvm::Function *F) {
  return F->hasFnAttribute(ShaderAttrKindStr);
}

void set_num_threads(llvm::Function *F, unsigned x, unsigned y, unsigned z) {
  const llvm::StringRef NumThreadsAttrKindStr = "hlsl.numthreads";
  std::string Str = llvm::formatv("{0},{1},{2}", x, y, z);
  F->addFnAttr(NumThreadsAttrKindStr, Str);
}

GlobalVariable *createGlobalVariableForResource(Module &M,
                                                const char *Name,
                                                llvm::Type *Ty) {
  auto *GV = new GlobalVariable(M, Ty, /*isConstant*/ false,
                                GlobalValue::LinkageTypes::ExternalLinkage,
                                /*Initializer*/ nullptr, Name);
  GV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::None);
  return GV;
}

std::vector<uint8_t> global_optimize_module(llvm::Module *module,
                                            const CompileConfig &config) {
  TI_AUTO_PROF
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_ERROR("Module broken");
  }

  for (llvm::Function &F : module->functions()) {
    if (directx12::is_cs_entry(&F))
      continue;
    F.addFnAttr(llvm::Attribute::AlwaysInline);
  }
  // FIXME: choose shader model based on feature used.
  llvm::StringRef triple = "dxil-pc-shadermodel6.0-compute";
  module->setTargetTriple(triple);
  module->setSourceFileName("");
  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  TargetOptions options;
  if (config.fast_math) {
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

  legacy::FunctionPassManager function_pass_manager(module);
  legacy::PassManager module_pass_manager;

  llvm::StringRef mcpu = "";
  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), mcpu.str(), "", options, llvm::Reloc::PIC_,
      llvm::CodeModel::Small,
      config.opt_level > 0 ? CodeGenOpt::Aggressive : CodeGenOpt::None));

  TI_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

  module->setDataLayout(target_machine->createDataLayout());

  // Lower taichi intrinsic first.
  module_pass_manager.add(createTaichiIntrinsicLowerPass(&config));

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
  // Add passes after inline.
  module_pass_manager.add(createTaichiRuntimeContextLowerPass());

  llvm::SmallString<256> str;
  llvm::raw_svector_ostream OS(str);
  // Write DXIL container to OS.
  target_machine->addPassesToEmitFile(module_pass_manager, OS, nullptr,
                                      CGFT_ObjectFile);

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
  if (config.print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer(
        "taichi_kernel_dx12_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (DX12)");
    writer.write(module);
  }
  return std::vector<uint8_t>(str.begin(), str.end());
}

}  // namespace directx12
}  // namespace taichi::lang
