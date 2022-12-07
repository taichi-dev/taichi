#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"
#include <chrono>


using namespace std::chrono_literals;


// -----------------------------------------------------------------------------

TEST_F(CapiTest, TestBehaviorCreateRuntime)
{
  TiRuntime runtime = ti_create_runtime(TiArch::TI_ARCH_MAX_ENUM);
  TiError error = ti_get_last_error(0, nullptr);

  TI_ASSERT(runtime == TI_NULL_HANDLE);
  TI_ASSERT(error == TI_ERROR_NOT_SUPPORTED);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);


  if(ti::is_arch_available(TI_ARCH_X64))
  {
    TiRuntime runtime = ti_create_runtime(TiArch::TI_ARCH_X64);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(runtime != TI_NULL_HANDLE);
    TI_ASSERT(error == TI_ERROR_NOT_SUPPORTED);
  }

  if(ti::is_arch_available(TiArch::TI_ARCH_VULKAN))
  {
    TiRuntime runtime = ti_create_runtime(TiArch::TI_ARCH_VULKAN);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(runtime != TI_NULL_HANDLE);
    TI_ASSERT(error == TI_ERROR_NOT_SUPPORTED);
  }

  if(ti::is_arch_available(TiArch::TI_ARCH_OPENGL))
  {
    TiRuntime runitme = ti_create_runtime(TiArch::TI_ARCH_OPENGL);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(runtime!=TI_NULL_HANDLE);
    TI_ASSERT(error==TI_ERROR_NOT_SUPPORTED);
  }

  if(ti::is_arch_available(TiArch::TI_ARCH_CUDA))
  {
    TiRuntime runtime = ti_create_runtime(TiArch::TI_ARCH_CUDA);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(runtime!=TI_NULL_HANDLE);
    TI_ASSERT(error==TI_ERROR_NOT_SUPPORTED);
  }
}


TEST_F(CapiTest, TestBehaviorDestroyRuntime)
{
  if(ti::is_arch_available(TiArch::TI_ARCH_VULKAN))
  {
    TiRuntime runtime = ti_create_runtime(TiArch::TI_ARCH_VULKAN);
    ti_destroy_runtime(runtime);
    TI_ASSERT(runtime==TI_NULL_HANDLE);
  }
  TiRuntime runtime = TI_NULL_HANDLE;
  ti_destroy_runtime(runtime);
  TiError error = ti_get_last_error(0,nullptr);

  TI_ASSERT(error==TI_ERROR_ARGUMENT_NULL);
  ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
}


TEST_F(CapiTest,TestBehaviorGetRuntimeCapabilities)
{
  uint32_t capability_count  = 0;
  TiCapabilityLevelInfo capabilities;
  if(ti::is_arch_available(TiArch::TI_ARCH_VULKAN))
  {
    TiRuntime runtime = ti_create_runtime(TiArch::TI_ARCH_VULKAN);
    ti_get_runtime_capabilities(runtime,&capability_count,&capabilities);
    TI_ASSERT(runtime!=TI_NULL_HANDLE);
    TI_ASSERT(capability_count!=0);
  }
  ti_get_runtime_capabilities(TI_NULL_HANDLE,&capability_count,&capabilities);
  TiError error = ti_get_last_error(0,nullptr);
  TI_ASSERT(error==TI_ERROR_ARGUMENT_NULL);
  ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
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
