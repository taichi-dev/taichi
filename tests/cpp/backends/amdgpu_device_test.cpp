#include "gtest/gtest.h"

#ifdef TI_WITH_AMDGPU
#include "taichi/ir/ir_builder.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#include "taichi/rhi/amdgpu/amdgpu_device.h"
#include "taichi/runtime/llvm/llvm_context_pass.h"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/IRReader/IRReader.h>

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
  const taichi::lang::DeviceAllocation device_alloc =
      device->allocate_memory(params);

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
  const taichi::lang::DeviceAllocation device_dest =
      device->allocate_memory(params);
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
    "define dso_local void @runtime_add(double* %0, double* %1, double* %2) #4 "
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
    "define dso_local void @runtime_add(double* %0, double* %1, double* %2) #4 "
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

}  // namespace lang
}  // namespace taichi
#endif
