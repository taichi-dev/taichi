#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

#define CHECK_TAICHI_SUCCESS()                      \
  {                                                 \
    TiError actual = ti_get_last_error(0, nullptr); \
    EXPECT_EQ(actual, TI_ERROR_SUCCESS);            \
  }

#define CHECK_TAICHI_ERROR_IS(expected)             \
  {                                                 \
    TiError actual = ti_get_last_error(0, nullptr); \
    EXPECT_EQ(actual, expected);                    \
    ti_set_last_error(TI_ERROR_SUCCESS, nullptr);   \
  }

// -----------------------------------------------------------------------------

TEST_F(CapiTest, TestBehaviorCreateRuntime) {
  auto inner = [](TiArch arch) {
    TiRuntime runtime = ti_create_runtime(arch);
    TI_ASSERT(runtime == TI_NULL_HANDLE);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_NOT_SUPPORTED);
  };

  // Attempt to create runtime for unknown arch.
  inner(TI_ARCH_MAX_ENUM);

  // Attempt to create runtime for unsupported archs.
  inner(TI_ARCH_JS);
  inner(TI_ARCH_CC);
  inner(TI_ARCH_WASM);
  inner(TI_ARCH_METAL);
  inner(TI_ARCH_DX11);
  inner(TI_ARCH_DX12);
  inner(TI_ARCH_OPENCL);
  inner(TI_ARCH_AMDGPU);
}

TEST_F(CapiTest, TestBehaviorDestroyRuntime) {
  // Attempt to destroy null handles.
  ti_destroy_runtime(TI_NULL_HANDLE);
  CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
}

TEST_F(CapiTest, TestBehaviorGetRuntimeCapabilities) {
  auto inner = [](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported so the test is skipped", arch);
      return;
    }

    TiRuntime runtime = ti_create_runtime(arch);

    {
      // Two nulls, considerred nop.
      ti_get_runtime_capabilities(runtime, nullptr, nullptr);
      CHECK_TAICHI_SUCCESS();
    }

    {
      // Count is not null, buffer is null.
      // Usually the case the user look up the number of capabilities will be
      // returned.
      uint32_t capability_count = 0;
      ti_get_runtime_capabilities(runtime, &capability_count, nullptr);
      CHECK_TAICHI_SUCCESS();
      switch (arch) {
        case TI_ARCH_VULKAN:
        case TI_ARCH_OPENGL:
          // Always have `TI_CAPABILITY_SPIRV_VERSION`.
          TI_ASSERT(capability_count > 0);
          break;
        default:
          TI_NOT_IMPLEMENTED;
      }
    }

    {
      // Count is null, buffer non-null.
      // Expect nop.
      std::vector<TiCapabilityLevelInfo> capabilities{
          TiCapabilityLevelInfo{
              (TiCapability)0xcbcbcbcb,
              0xcbcbcbcb,
          },
      };
      ti_get_runtime_capabilities(runtime, nullptr, capabilities.data());
      CHECK_TAICHI_SUCCESS();
      EXPECT_EQ(capabilities.at(0).capability, (TiCapability)0xcbcbcbcb);
      EXPECT_EQ(capabilities.at(0).level, 0xcbcbcbcb);
    }

    {
      // Both non-null.
      // Normal usage.
      uint32_t capability_count = 0;
      ti_get_runtime_capabilities(runtime, &capability_count, nullptr);
      CHECK_TAICHI_SUCCESS();
      std::vector<TiCapabilityLevelInfo> capabilities(capability_count);
      ti_get_runtime_capabilities(runtime, &capability_count,
                                  capabilities.data());
      CHECK_TAICHI_SUCCESS();
      for (size_t i = 0; i < capability_count; ++i) {
        TI_ASSERT(capabilities.at(i).capability !=
                  (TiCapability)TI_CAPABILITY_RESERVED);
        TI_ASSERT(capabilities.at(i).level != 0);
      }
    }
  };

  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorAllocateMemory)
{
  TiError error = TI_ERROR_SUCCESS;

 
  TiMemoryAllocateInfo* allocate_info = new TiMemoryAllocateInfo;
  allocate_info->size = 1024;
  if(ti::is_arch_available(TI_ARCH_VULKAN))
  {
    TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
    for(int i = 0;i<4;++i)
    {
      allocate_info->usage = TI_MEMORY_USAGE_STORAGE_BIT<<i;
      TiMemory memory = ti_allocate_memory(runtime,allocate_info);
      TI_ASSERT(memory!=TI_NULL_HANDLE);
    }

    allocate_info->size = 1000000000000000000;
    ti_allocate_memory(runtime,allocate_info);
    error = ti_get_last_error(0,nullptr);
    //std::cout<<error<<std::endl;
    TI_ASSERT(error==TI_ERROR_OUT_OF_MEMORY);                   
    allocate_info->size = 1024;
  }
  allocate_info->size = 1024;
   
  if(ti::is_arch_available(TI_ARCH_OPENGL))
  {
    TiRuntime runtime = ti_create_runtime(TI_ARCH_OPENGL);
    for(int i = 0;i<4;++i)
    {
      allocate_info->usage = TI_MEMORY_USAGE_STORAGE_BIT<<i;
      TiMemory memory = ti_allocate_memory(runtime,allocate_info);
      TI_ASSERT(memory!=TI_NULL_HANDLE);
    }
    allocate_info->size = 1000000000000000000;
    ti_allocate_memory(runtime,allocate_info);
    error = ti_get_last_error(0,nullptr);
    //std::cout<<error<<std::endl;
    TI_ASSERT(error==TI_ERROR_OUT_OF_MEMORY);                   
    allocate_info->size = 1024;
  }

  ti_allocate_memory(TI_NULL_HANDLE,nullptr);
  error = ti_get_last_error(0,nullptr);
  TI_ASSERT(error==TI_ERROR_ARGUMENT_NULL);
  ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
}

TEST_F(CapiTest, TestBehaviorFreeMemory)    
{
  if(ti::is_arch_available(TI_ARCH_VULKAN))
  {
    TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
    TiMemoryAllocateInfo* allocate_info = new TiMemoryAllocateInfo;
    allocate_info->size = 1024;
    allocate_info->usage = TI_MEMORY_USAGE_STORAGE_BIT;
    TiMemory memory = ti_allocate_memory(runtime,allocate_info);
    ti_free_memory(runtime,memory);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
  }
}

TEST_F (CapiTest, TestBehaviorMapMemory)
{
  TiMemoryAllocateInfo* allocate_info = new TiMemoryAllocateInfo;
  allocate_info->size = 1024;
  allocate_info->usage = TI_MEMORY_USAGE_STORAGE_BIT;
  if(ti::is_arch_available(TI_ARCH_VULKAN))
  {
    TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
    TiMemory memory = ti_allocate_memory(runtime,allocate_info);
    ti_map_memory(runtime,memory);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
  }
   if(ti::is_arch_available(TI_ARCH_CUDA))
  {
    TiRuntime runtime = ti_create_runtime(TI_ARCH_CUDA);
    TiMemory memory = ti_allocate_memory(runtime,allocate_info);
    ti_map_memory(runtime,memory);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
  }
   if(ti::is_arch_available(TI_ARCH_X64))
  {
    TiRuntime runtime = ti_create_runtime(TI_ARCH_X64);
    TiMemory memory = ti_allocate_memory(runtime,allocate_info);
    ti_map_memory(runtime,memory);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
  }
   if(ti::is_arch_available(TI_ARCH_OPENGL))
  {
    TiRuntime runtime = ti_create_runtime(TI_ARCH_OPENGL);
    TiMemory memory = ti_allocate_memory(runtime,allocate_info);
    ti_map_memory(runtime,memory);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
  }
  ti_map_memory(TI_NULL_HANDLE,TI_NULL_HANDLE);
  TiError error = ti_get_last_error(0,nullptr);
  TI_ASSERT(error,TI_ERROR_ARGUMENT_NULL);
  ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
}
