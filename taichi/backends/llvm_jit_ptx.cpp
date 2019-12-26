#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#if defined(TLANG_WITH_CUDA)
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif
#include "cuda_context.h"
#include "llvm_jit.h"
#include <taichi/program.h>
#include <taichi/context.h>
#include <taichi/system/timer.h>

TLANG_NAMESPACE_BEGIN

#if defined(TLANG_WITH_CUDA)

std::string cuda_mattrs() {
  return "+ptx50";
}

std::unique_ptr<CUDAContext> cuda_context;  // TODO:..

std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module) {
  // Part of this function is borrowed from Halide::CodeGen_PTX_Dev.cpp
  using namespace llvm;

  llvm::Triple triple(module->getTargetTriple());

  // Allocate target machine

  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TC_ERROR_UNLESS(target, err_str);

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
      triple.str(), cuda_context->get_mcpu(), cuda_mattrs(), options,
      llvm::Reloc::PIC_, llvm::CodeModel::Small, CodeGenOpt::Aggressive));

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

  std::string buffer(outstr.begin(), outstr.end());

  // Null-terminate the ptx source
  buffer.push_back(0);
  return buffer;
}

#define checkCudaErrors(err)                                \
  if ((err) != CUDA_SUCCESS)                                \
    TC_ERROR("Cuda Error {}: {}", get_cuda_error_name(err), \
             get_cuda_error_string(err));

std::string get_cuda_error_name(CUresult err) {
  const char *ptr;
  cuGetErrorName(err, &ptr);
  return std::string(ptr);
}

std::string get_cuda_error_string(CUresult err) {
  const char *ptr;
  cuGetErrorString(err, &ptr);
  return std::string(ptr);
}

CUDAContext::CUDAContext() {
  // CUDA initialization
  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGetCount(&devCount));
  checkCudaErrors(cuDeviceGet(&device, 0));

  char name[128];
  checkCudaErrors(cuDeviceGetName(name, 128, device));
  std::cout << "Using CUDA Device [0]: " << name << "\n";

  int devMajor, devMinor;
  checkCudaErrors(cuDeviceComputeCapability(&devMajor, &devMinor, device));
  std::cout << "Device Compute Capability: " << devMajor << "." << devMinor
            << "\n";
  if (devMajor < 2) {
    TC_ERROR("Device 0 is not SM 2.0 or greater");
  }
  // Create driver context
  checkCudaErrors(cuCtxCreate(&context, 0, device));
  checkCudaErrors(cuMemAlloc(&context_buffer, sizeof(Context)));

  int cap_major, cap_minor;
  cudaDeviceGetAttribute(&cap_major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&cap_minor, cudaDevAttrComputeCapabilityMinor, 0);
  mcpu = fmt::format("sm_{}{}", cap_major, cap_minor);
}

CUmodule CUDAContext::compile(const std::string &ptx) {
  cuda_context->make_current();
  // Create module for object
  CUmodule cudaModule;
  TC_INFO("PTX size: {:.2f}KB", ptx.size() / 1024.0);
  // auto t = Time::get_time();
  checkCudaErrors(cuModuleLoadDataEx(&cudaModule, ptx.c_str(), 0, 0, 0));
  // TC_INFO("CUDA module load time : {}ms", (Time::get_time() - t) * 1000);
  cudaModules.push_back(cudaModule);
  return cudaModule;
}

CUfunction CUDAContext::get_function(CUmodule module,
                                     const std::string &func_name) {
  cuda_context->make_current();
  CUfunction func;
  // auto t = Time::get_time();
  checkCudaErrors(cuModuleGetFunction(&func, module, func_name.c_str()));
  // t = Time::get_time() - t;
  // TC_INFO("Kernel {} compilation time: {}ms", func_name, t * 1000);
  return func;
}

void CUDAContext::launch(CUfunction func,
                         void *context_ptr,
                         unsigned gridDim,
                         unsigned blockDim) {
  cuda_context->make_current();
  // Kernel parameters

  checkCudaErrors(cuMemcpyHtoD(context_buffer, context_ptr, sizeof(Context)));

  void *KernelParams[] = {&context_buffer};

  // Kernel launch
  TC_WARN_IF(
      gridDim * blockDim > 1024 * 1024,
      "random number generator only initialized for 1024 * 1024 threads.");
  if (gridDim > 0) {
    checkCudaErrors(cuLaunchKernel(func, gridDim, 1, 1, blockDim, 1, 1, 0,
                                   nullptr, KernelParams, nullptr));
  }

  if (get_current_program().config.debug) {
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err) {
      TC_ERROR("CUDA Kernel Launch Error: {}", cudaGetErrorString(err));
    }
  }
}

CUDAContext::~CUDAContext() {
  /*
  checkCudaErrors(cuMemFree(context_buffer));
  for (auto cudaModule: cudaModules)
    checkCudaErrors(cuModuleUnload(cudaModule));
  checkCudaErrors(cuCtxDestroy(context));
  */
}

#else
std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module) {
  TC_NOT_IMPLEMENTED
}

int compile_ptx_and_launch(const std::string &ptx,
                           const std::string &kernel_name,
                           void *) {
  TC_NOT_IMPLEMENTED
}
#endif

TLANG_NAMESPACE_END
