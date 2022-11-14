#include "gtest/gtest.h"

#ifdef TI_WITH_AMDGPU
#include "taichi/ir/ir_builder.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#include "taichi/rhi/amdgpu/amdgpu_device.h"
#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {
TEST(AMDGPU, CreateDeviceAndAlloc) {
    std::unique_ptr<amdgpu::AmdgpuDevice> device =
        std::make_unique<amdgpu::AmdgpuDevice>();
    EXPECT_TRUE(device != nullptr);
    taichi::lang::Device::AllocParams params;
    params.size = 1048576;
    params.host_read = false;
    params.host_write = false;
    const taichi::lang::DeviceAllocation device_alloc =
       device->allocate_memory(params);
    
    // The purpose of the device_alloc_guard is to rule out double free
    const taichi::lang::DeviceAllocationGuard device_alloc_guard(device_alloc);
    // Map to CPU, write some values, then check those values
    void *mapped = device->map(device_alloc);
    int *mapped_int = reinterpret_cast<int *>(mapped);
    for (int i = 0; i < 100; i++) {
        mapped_int[i] = i;
    }
    device->unmap(device_alloc);

    mapped = device->map(device_alloc);
    mapped_int = reinterpret_cast<int *>(mapped);
    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(mapped_int[i], i);
    }
    device->unmap(device_alloc);
}

TEST(AMDGPU, ImportMemory) {
    std::unique_ptr<amdgpu::AmdgpuDevice> device =
    std::make_unique<amdgpu::AmdgpuDevice>();
    EXPECT_TRUE(device != nullptr);

    int *ptr = nullptr;
    AMDGPUDriver::get_instance().malloc_managed(
        (void **)&ptr, 400, HIP_MEM_ATTACH_GLOBAL);
    const taichi::lang::DeviceAllocation device_alloc =
       device->import_memory(ptr, 400);

    for (int i = 0; i < 100; i++) {
        ptr[i] = i;
    }

    taichi::lang::Device::AllocParams params;
    params.size = 400;
    params.host_read = false;
    params.host_write = false;
    const taichi::lang::DeviceAllocation device_dest =
       device->allocate_memory(params);
    const taichi::lang::DeviceAllocationGuard device_dest_guard(device_dest);

    AMDGPUDriver::get_instance().stream_synchronize(nullptr);
    device->memcpy_internal(device_dest.get_ptr(0), 
                            device_alloc.get_ptr(0), 400);
    void *mapped = device->map(device_dest);
    int *mapped_int = reinterpret_cast<int *>(mapped);

    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(mapped_int[i], i);
    }
    device->unmap(device_dest);
    // import memory should been deallocated manually
    AMDGPUDriver::get_instance().mem_free(ptr);
}

TEST(AMDGPU, CreateContextAndLaunchKernel) {
    // NOT_IMPLEMENTED
}

TEST(AMDGPU, FetchResult) {
    // NOT_IMPLEMENTED
}

} // lang
} // taichi
#endif