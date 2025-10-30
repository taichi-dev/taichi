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

// === CHANGED SECTION: HEADER INCLUDES ===
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
// #include "llvm/Support/Host.h" // This was not used, but good to be aware of.
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
// #include "llvm/IR/LegacyPassManager.h" // Obsolete: Removed
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Target/TargetMachine.h"
// #include "llvm/Transforms/IPO/PassManagerBuilder.h" // Obsolete: Removed
// #include "llvm/Transforms/InstCombine/InstCombine.h" // Included via PassBuilder
// #include "llvm/Transforms/Scalar.h" // Included via PassBuilder
// #include "llvm/Transforms/Scalar/GVN.h" // Included via PassBuilder
// #include "llvm/Transforms/IPO.h" // Included via PassBuilder
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/GlobalVariable.h"
// New includes for the New Pass Manager (NPM)
#include "llvm/Passes/PassBuilder.h"
// === END OF CHANGED SECTION ===


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

// === CHANGED SECTION: ENTIRE FUNCTION REWRITTEN ===
// The `global_optimize_module` function has been completely rewritten to use the
// New Pass Manager (NPM) instead of the removed Legacy Pass Manager (LPM).
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
    // Mark other functions for inlining.
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
    options.UnsafeFPMath = true;
    options.NoInfsFPMath = true;
    options.NoNaNsFPMath = true;
  }
  options.HonorSignDependentRoundingFPMathOption = false;
  options.NoZerosInBSS = false;
  options.GuaranteedTailCallOpt = false;

  llvm::StringRef mcpu = "";
  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), mcpu.str(), "", options, llvm::Reloc::PIC_,
      llvm::CodeModel::Small,
      config.opt_level > 0 ? CodeGenOpt::Aggressive : CodeGenOpt::None));

  TI_ERROR_UNLESS(target_machine, "Could not allocate target machine!");

  module->setDataLayout(target_machine->createDataLayout());

  // === New Pass Manager Setup ===
  // 1. Create the analysis managers.
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  // 2. Create the PassBuilder.
  llvm::PassBuilder PB(target_machine.get());

  // 3. Register all the standard analyses.
  FAM.registerPass([&] { return target_machine->getTargetIRAnalysis(); });
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // 4. Create the main pass manager.
  llvm::ModulePassManager MPM;
  
  // Lower taichi intrinsic first. This is a custom pass.
  MPM.addPass(createTaichiIntrinsicLowerPass(&config));
  
  // 5. Build the default optimization pipeline for O3.
  llvm::PassBuilder::OptimizationLevel opt_level = llvm::PassBuilder::OptimizationLevel::O3;
  // This will add inlining, vectorization, etc., replacing `PassManagerBuilder`.
  // Note: We are now creating a more complex pipeline. We can use `buildPerModuleDefaultPipeline`
  // but to insert passes in the middle, we construct it manually. A simpler way is to
  // use `parsePassPipeline`. For now, we build the default pipeline first.
  if (config.opt_level > 0) {
      MPM = PB.buildPerModuleDefaultPipeline(opt_level);
  }

  // Add the second custom pass, which should run after inlining.
  MPM.addPass(createTaichiRuntimeContextLowerPass());

  llvm::SmallString<0> str;
  llvm::raw_svector_ostream OS(str);
  
  // 6. Add the pass to emit the object file to the stream.
  if (auto err = target_machine->addPassesToEmitFile(MPM, OS, nullptr, CGFT_ObjectFile)) {
    TI_ERROR("Failed to addPassesToEmitFile");
  }

  // 7. Run the entire pipeline.
  {
    TI_PROFILER("llvm_module_pass");
    MPM.run(*module, MAM);
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