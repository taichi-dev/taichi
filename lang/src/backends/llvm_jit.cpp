#if defined(TLANG_WITH_LLVM)
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#if defined(TLANG_WITH_CUDA)
#include <cuda.h>
#endif
#include "llvm_jit.h"

TLANG_NAMESPACE_BEGIN

// This file is based on Halide:CodeGen_PTX_Dev.cpp

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
#if defined(TLANG_WITH_CUDA)
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

#define checkCudaErrors(err) \
  if ((err) != CUDA_SUCCESS) \
    TC_ERROR("Cuda Error {}", get_error_name(err));

// from Halide::Runtime::Internal::Cuda
const char *get_error_name(CUresult err) {
  switch (err) {
    case CUDA_SUCCESS:
      return "CUDA_SUCCESS";
    case CUDA_ERROR_INVALID_VALUE:
      return "CUDA_ERROR_INVALID_VALUE";
    case CUDA_ERROR_OUT_OF_MEMORY:
      return "CUDA_ERROR_OUT_OF_MEMORY";
    case CUDA_ERROR_NOT_INITIALIZED:
      return "CUDA_ERROR_NOT_INITIALIZED";
    case CUDA_ERROR_DEINITIALIZED:
      return "CUDA_ERROR_DEINITIALIZED";
    case CUDA_ERROR_PROFILER_DISABLED:
      return "CUDA_ERROR_PROFILER_DISABLED";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
      return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
      return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
      return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
    case CUDA_ERROR_NO_DEVICE:
      return "CUDA_ERROR_NO_DEVICE";
    case CUDA_ERROR_INVALID_DEVICE:
      return "CUDA_ERROR_INVALID_DEVICE";
    case CUDA_ERROR_INVALID_IMAGE:
      return "CUDA_ERROR_INVALID_IMAGE";
    case CUDA_ERROR_INVALID_CONTEXT:
      return "CUDA_ERROR_INVALID_CONTEXT";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
      return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
    case CUDA_ERROR_MAP_FAILED:
      return "CUDA_ERROR_MAP_FAILED";
    case CUDA_ERROR_UNMAP_FAILED:
      return "CUDA_ERROR_UNMAP_FAILED";
    case CUDA_ERROR_ARRAY_IS_MAPPED:
      return "CUDA_ERROR_ARRAY_IS_MAPPED";
    case CUDA_ERROR_ALREADY_MAPPED:
      return "CUDA_ERROR_ALREADY_MAPPED";
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
      return "CUDA_ERROR_NO_BINARY_FOR_GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED:
      return "CUDA_ERROR_ALREADY_ACQUIRED";
    case CUDA_ERROR_NOT_MAPPED:
      return "CUDA_ERROR_NOT_MAPPED";
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
      return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
      return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
    case CUDA_ERROR_ECC_UNCORRECTABLE:
      return "CUDA_ERROR_ECC_UNCORRECTABLE";
    case CUDA_ERROR_UNSUPPORTED_LIMIT:
      return "CUDA_ERROR_UNSUPPORTED_LIMIT";
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
      return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
      return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
    case CUDA_ERROR_INVALID_PTX:
      return "CUDA_ERROR_INVALID_PTX";
    case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
      return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
    case CUDA_ERROR_NVLINK_UNCORRECTABLE:
      return "CUDA_ERROR_NVLINK_UNCORRECTABLE";
    case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
      return "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";
    case CUDA_ERROR_INVALID_SOURCE:
      return "CUDA_ERROR_INVALID_SOURCE";
    case CUDA_ERROR_FILE_NOT_FOUND:
      return "CUDA_ERROR_FILE_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
      return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
      return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
    case CUDA_ERROR_OPERATING_SYSTEM:
      return "CUDA_ERROR_OPERATING_SYSTEM";
    case CUDA_ERROR_INVALID_HANDLE:
      return "CUDA_ERROR_INVALID_HANDLE";
    case CUDA_ERROR_NOT_FOUND:
      return "CUDA_ERROR_NOT_FOUND";
    case CUDA_ERROR_NOT_READY:
      return "CUDA_ERROR_NOT_READY";
    case CUDA_ERROR_ILLEGAL_ADDRESS:
      return "CUDA_ERROR_ILLEGAL_ADDRESS";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
    case CUDA_ERROR_LAUNCH_TIMEOUT:
      return "CUDA_ERROR_LAUNCH_TIMEOUT";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
      return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
      return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
      return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
      return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:
      return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
      // A trap instruction produces the below error, which is how we codegen
      // asserts on GPU
    case CUDA_ERROR_ILLEGAL_INSTRUCTION:
      return "Illegal instruction or Halide assertion failure inside kernel";
    case CUDA_ERROR_MISALIGNED_ADDRESS:
      return "CUDA_ERROR_MISALIGNED_ADDRESS";
    case CUDA_ERROR_INVALID_ADDRESS_SPACE:
      return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
    case CUDA_ERROR_INVALID_PC:
      return "CUDA_ERROR_INVALID_PC";
    case CUDA_ERROR_LAUNCH_FAILED:
      return "CUDA_ERROR_LAUNCH_FAILED";
    case CUDA_ERROR_NOT_PERMITTED:
      return "CUDA_ERROR_NOT_PERMITTED";
    case CUDA_ERROR_NOT_SUPPORTED:
      return "CUDA_ERROR_NOT_SUPPORTED";
    case CUDA_ERROR_UNKNOWN:
      return "CUDA_ERROR_UNKNOWN";
    default:
      // This is unfortunate as usually get_cuda_error is called in the middle
      // of an error print, but dropping the number on the floor is worse.
      TC_ERROR("Unknown cuda error {}", int(err));
      return "<Unknown error>";
  }
}

int compile_ptx_and_launch(const std::string &ptx,
                           const std::string &kernel_name) {
  CUdevice device;
  CUmodule cudaModule;
  CUcontext context;
  CUfunction function;
  CUlinkState linker;
  int devCount;

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
    std::cerr << "ERROR: Device 0 is not SM 2.0 or greater\n";
    return 1;
  }

  std::string str = ptx;

  TC_TRACE("Compiling PTX: \n{}", str);

  // Create driver context
  checkCudaErrors(cuCtxCreate(&context, 0, device));

  // Create module for object
  checkCudaErrors(cuModuleLoadDataEx(&cudaModule, str.c_str(), 0, 0, 0));

  TC_TRACE("Loading symbol {}", kernel_name);
  // Get kernel function
  checkCudaErrors(
      cuModuleGetFunction(&function, cudaModule, kernel_name.c_str()));

  // Device data
  CUdeviceptr devBufferA;
  CUdeviceptr devBufferB;
  CUdeviceptr devBufferC;

  checkCudaErrors(cuMemAlloc(&devBufferA, sizeof(float) * 16));
  checkCudaErrors(cuMemAlloc(&devBufferB, sizeof(float) * 16));
  checkCudaErrors(cuMemAlloc(&devBufferC, sizeof(float) * 16));

  float *hostA = new float[16];
  float *hostB = new float[16];
  float *hostC = new float[16];

  // Populate input
  for (unsigned i = 0; i != 16; ++i) {
    hostA[i] = (float)i;
    hostB[i] = (float)(2 * i);
    hostC[i] = 0.0f;
  }

  checkCudaErrors(cuMemcpyHtoD(devBufferA, &hostA[0], sizeof(float) * 16));
  checkCudaErrors(cuMemcpyHtoD(devBufferB, &hostB[0], sizeof(float) * 16));

  unsigned blockSizeX = 16;
  unsigned blockSizeY = 1;
  unsigned blockSizeZ = 1;
  unsigned gridSizeX = 1;
  unsigned gridSizeY = 1;
  unsigned gridSizeZ = 1;

  // Kernel parameters
  void *KernelParams[] = {&devBufferA, &devBufferB, &devBufferC};

  std::cout << "Launching kernel\n";

  // Kernel launch
  checkCudaErrors(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
                                 blockSizeX, blockSizeY, blockSizeZ, 0, NULL,
                                 KernelParams, NULL));

  // Retrieve device data
  checkCudaErrors(cuMemcpyDtoH(&hostC[0], devBufferC, sizeof(float) * 16));

  std::cout << "Results:\n";
  for (unsigned i = 0; i != 16; ++i) {
    std::cout << hostA[i] << " + " << hostB[i] << " = " << hostC[i] << "\n";
  }

  // Clean up after ourselves
  delete[] hostA;
  delete[] hostB;
  delete[] hostC;

  // Clean-up
  checkCudaErrors(cuMemFree(devBufferA));
  checkCudaErrors(cuMemFree(devBufferB));
  checkCudaErrors(cuMemFree(devBufferC));
  checkCudaErrors(cuModuleUnload(cudaModule));
  checkCudaErrors(cuCtxDestroy(context));
}
#else
std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module) {
  TC_NOT_IMPLEMENTED
}

int compile_ptx_and_launch(const std::string &ptx,
                           const std::string &kernel_name) {
  TC_NOT_IMPLEMENTED
}
#endif

TLANG_NAMESPACE_END
#endif