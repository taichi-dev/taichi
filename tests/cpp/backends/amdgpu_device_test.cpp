#include "gtest/gtest.h"

#ifdef TI_WITH_AMDGPU
#include "taichi/ir/ir_builder.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#include "taichi/rhi/amdgpu/amdgpu_device.h"
#include "taichi/runtime/amdgpu/jit_amdgpu.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/runtime/llvm/llvm_context_pass.h"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {

TEST(AMDGPU, CreateDeviceAndAlloc) {
  std::unique_ptr<amdgpu::AmdgpuDevice> device =
      std::make_unique<amdgpu::AmdgpuDevice>();
  EXPECT_TRUE(device != nullptr);
  taichi::lang::Device::AllocParams params;
  params.size = 400;
  params.host_read = true;
  params.host_write = true;
  taichi::lang::DeviceAllocation device_alloc;
  EXPECT_EQ(device->allocate_memory(params, &device_alloc), RhiResult::success);

  // The purpose of the device_alloc_guard is to rule out double free
  const taichi::lang::DeviceAllocationGuard device_alloc_guard(device_alloc);
  // Map to CPU, write some values, then check those values
  void *mapped;
  EXPECT_EQ(device->map(device_alloc, &mapped), RhiResult::success);

  int *mapped_int = reinterpret_cast<int *>(mapped);
  for (int i = 0; i < params.size / sizeof(int); i++) {
    mapped_int[i] = i;
  }
  device->unmap(device_alloc);
  EXPECT_EQ(device->map(device_alloc, &mapped), RhiResult::success);

  mapped_int = reinterpret_cast<int *>(mapped);
  for (int i = 0; i < params.size / sizeof(int); i++) {
    EXPECT_EQ(mapped_int[i], i);
  }
  device->unmap(device_alloc);
}

TEST(AMDGPU, ImportMemory) {
  std::unique_ptr<amdgpu::AmdgpuDevice> device =
      std::make_unique<amdgpu::AmdgpuDevice>();
  EXPECT_TRUE(device != nullptr);

  int *ptr = nullptr;
  size_t mem_size = 400;
  AMDGPUDriver::get_instance().malloc_managed((void **)&ptr, mem_size,
                                              HIP_MEM_ATTACH_GLOBAL);
  const taichi::lang::DeviceAllocation device_alloc =
      device->import_memory(ptr, mem_size);

  for (int i = 0; i < mem_size / sizeof(int); i++) {
    ptr[i] = i;
  }

  taichi::lang::Device::AllocParams params;
  params.size = 400;
  params.host_read = true;
  params.host_write = true;
  taichi::lang::DeviceAllocation device_dest;
  EXPECT_EQ(device->allocate_memory(params, &device_dest), RhiResult::success);
  const taichi::lang::DeviceAllocationGuard device_dest_guard(device_dest);

  AMDGPUDriver::get_instance().stream_synchronize(nullptr);
  device->memcpy_internal(device_dest.get_ptr(0), device_alloc.get_ptr(0),
                          params.size);
  void *mapped;
  EXPECT_EQ(device->map(device_dest, &mapped), RhiResult::success);

  int *mapped_int = reinterpret_cast<int *>(mapped);

  for (int i = 0; i < params.size / sizeof(int); i++) {
    EXPECT_EQ(mapped_int[i], i);
  }
  device->unmap(device_dest);
  // import memory should been deallocated manually
  AMDGPUDriver::get_instance().mem_free(ptr);
}

TEST(AMDGPU, CreateContextAndGetMemInfo) {
  auto total_size = AMDGPUContext::get_instance().get_total_memory();
  auto free_size = AMDGPUContext::get_instance().get_free_memory();
  EXPECT_GE(total_size, free_size);
  EXPECT_GE(free_size, 0);
}

TEST(AMDGPU, ConvertAllocaInstAddressSpacePass) {
  const std::string program =
      "define dso_local void @runtime_add(double* %0, double* %1, double* %2) "
      "#4 "
      "{\n"
      "%4 = alloca double*, align 8\n"
      "%5 = alloca double*, align 8\n"
      "%6 = alloca double*, align 8\n"
      "store double* %0, double** %4, align 8\n"
      "store double* %1, double** %5, align 8\n"
      "store double* %2, double** %6, align 8\n"
      "%7 = load double*, double** %4, align 8\n"
      "%8 = load double, double* %7, align 8\n"
      "%9 = load double*, double** %5, align 8\n"
      "%10 = load double, double* %9, align 8\n"
      "%11 = fadd contract double %8, %10\n"
      "%12 = load double*, double** %6, align 8\n"
      "store double %11, double* %12, align 8\n"
      "ret void\n"
      "}\n";
  llvm::LLVMContext llvm_context;
  llvm::SMDiagnostic diagnostic_err;
  std::unique_ptr<llvm::Module> llvm_module = llvm::parseIR(
      llvm::MemoryBuffer::getMemBuffer(program)->getMemBufferRef(),
      diagnostic_err, llvm_context);
  llvm::legacy::FunctionPassManager function_pass_manager(llvm_module.get());
  function_pass_manager.add(new AMDGPUConvertAllocaInstAddressSpacePass());
  function_pass_manager.doInitialization();
  for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func) {
    function_pass_manager.run(*func);
  }
  function_pass_manager.doFinalization();
  auto func = llvm_module->getFunction("runtime_add");
  for (auto &bb : *func) {
    for (llvm::Instruction &inst : bb) {
      auto alloca_inst = llvm::dyn_cast<AllocaInst>(&inst);
      if (!alloca_inst)
        continue;
      EXPECT_EQ(alloca_inst->getAddressSpace(), 5);
    }
    int cast_num = 0;
    for (llvm::Instruction &inst : bb) {
      auto cast_inst = llvm::dyn_cast<AddrSpaceCastInst>(&inst);
      if (!cast_inst)
        continue;
      cast_num++;
    }
    EXPECT_EQ(cast_num, 3);
  }
}

TEST(AMDGPU, ConvertFuncParamAddressSpacePass) {
  const std::string program =
      "define dso_local void @runtime_add(double* %0, double* %1, double* %2) "
      "#4 "
      "{\n"
      "%4 = alloca double*, align 8\n"
      "%5 = alloca double*, align 8\n"
      "%6 = alloca double*, align 8\n"
      "store double* %0, double** %4, align 8\n"
      "store double* %1, double** %5, align 8\n"
      "store double* %2, double** %6, align 8\n"
      "%7 = load double*, double** %4, align 8\n"
      "%8 = load double, double* %7, align 8\n"
      "%9 = load double*, double** %5, align 8\n"
      "%10 = load double, double* %9, align 8\n"
      "%11 = fadd contract double %8, %10\n"
      "%12 = load double*, double** %6, align 8\n"
      "store double %11, double* %12, align 8\n"
      "ret void\n"
      "}\n";
  llvm::LLVMContext llvm_context;
  llvm::SMDiagnostic diagnostic_err;
  std::unique_ptr<llvm::Module> llvm_module = llvm::parseIR(
      llvm::MemoryBuffer::getMemBuffer(program)->getMemBufferRef(),
      diagnostic_err, llvm_context);
  llvm::legacy::PassManager module_pass_manager;
  module_pass_manager.add(new AMDGPUConvertFuncParamAddressSpacePass());
  module_pass_manager.run(*llvm_module);
  auto func = llvm_module->getFunction("runtime_add");
  for (llvm::Function::arg_iterator I = func->arg_begin(), E = func->arg_end();
       I != E; ++I) {
    if (I->getType()->getTypeID() == llvm::Type::PointerTyID) {
      EXPECT_EQ(I->getType()->getPointerAddressSpace(), 1);
    }
  }
}

TEST(AMDGPU, CompileProgramAndLaunch) {
  std::string program =
      "target datalayout = "
      "\"e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:"
      "64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:"
      "1024-v2048:2048-n32:64-S32-A5-G1-ni:7\"\n"
      "target triple = \"amdgcn-amd-amdhsa\"\n"
      "define amdgpu_kernel void @runtime_add(double addrspace(1)* %0, double "
      "addrspace(1)* %1, double addrspace(1)* %2) #0 {\n"
      "  %4 = alloca double*, align 8, addrspace(5)\n"
      "  %5 = addrspacecast double addrspace(1)* %2 to double*\n"
      "  %6 = addrspacecast double addrspace(1)* %1 to double*\n"
      "  %7 = addrspacecast double addrspace(1)* %0 to double*\n"
      "  %8 = addrspacecast double* addrspace(5)* %4 to double**\n"
      "  %9 = alloca double*, align 8, addrspace(5)\n"
      "  %10 = addrspacecast double* addrspace(5)* %9 to double**\n"
      "  %11 = alloca double*, align 8, addrspace(5)\n"
      "  %12 = addrspacecast double* addrspace(5)* %11 to double**\n"
      "  store double* %7, double** %8, align 8\n"
      "  store double* %6, double** %10, align 8\n"
      "  store double* %5, double** %12, align 8\n"
      "  %13 = load double*, double** %8, align 8\n"
      "  %14 = load double, double* %13, align 8\n"
      "  %15 = load double*, double** %10, align 8\n"
      "  %16 = load double, double* %15, align 8\n"
      "  %17 = fadd contract double %14, %16\n"
      "  %18 = load double*, double** %12, align 8\n"
      "  store double %17, double* %18, align 8\n"
      "  ret void\n"
      "}\n";
  llvm::LLVMContext llvm_context;
  llvm::SMDiagnostic diagnostic_err;
  std::unique_ptr<llvm::Module> llvm_module = llvm::parseIR(
      llvm::MemoryBuffer::getMemBuffer(program)->getMemBufferRef(),
      diagnostic_err, llvm_context);

  // auto amdgpu_session = new JITSessionAMDGPU(new TaichiLLVMContext(new
  // CompileConfig, Arch::amdgpu), new CompileConfig(), llvm::DataLayout(""));
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUAsmParser();
  auto amdgpu_session = new JITSessionAMDGPU(nullptr, default_compile_config,
                                             llvm::DataLayout(""));
  auto amdgpu_module = amdgpu_session->add_module(std::move(llvm_module), 0);
  std::vector<void *> arg_pointers;
  std::vector<int> arg_sizes;
  double *args[3];
  size_t arg_size = sizeof(double);
  AMDGPUDriver::get_instance().malloc((void **)&(args[0]), sizeof(double) * 3);
  args[1] = args[0] + 1;
  args[2] = args[0] + 2;
  double a = 10.0;
  double b = 7.0;
  double ret;
  AMDGPUDriver::get_instance().memcpy_host_to_device(args[0], &a,
                                                     sizeof(double));
  AMDGPUDriver::get_instance().memcpy_host_to_device(args[1], &b,
                                                     sizeof(double));
  arg_pointers.push_back((void *)&args[0]);
  arg_pointers.push_back((void *)&args[1]);
  arg_pointers.push_back((void *)&args[2]);
  arg_sizes.push_back(arg_size);
  arg_sizes.push_back(arg_size);
  arg_sizes.push_back(arg_size);
  amdgpu_module->call("runtime_add", arg_pointers, arg_sizes);
  AMDGPUDriver::get_instance().stream_synchronize(nullptr);
  AMDGPUDriver::get_instance().memcpy_device_to_host(&ret, args[2],
                                                     sizeof(double));
  EXPECT_EQ(ret, 17);
  AMDGPUDriver::get_instance().mem_free(args[0]);
}

}  // namespace lang
}  // namespace taichi
#endif
