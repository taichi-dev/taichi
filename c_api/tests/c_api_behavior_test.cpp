#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"
#include <chrono>


using namespace std::chrono_literals;


// -----------------------------------------------------------------------------

TEST_F(CapiTest, TestBehaviorCreateRuntime)
{
  TiError error = TI_ERROR_SUCCESS;

  TiRuntime runtime = ti_create_runtime(TiArch::TI_ARCH_MAX_ENUM);
  error = ti_get_last_error(0, nullptr);
  std::cout<<error<<std::endl;
  TI_ASSERT(runtime == TI_NULL_HANDLE);
  TI_ASSERT(error == TI_ERROR_NOT_SUPPORTED);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);

  if(ti::is_arch_available(TiArch::TI_ARCH_VULKAN))
  {
    std::cout<<"vulkan"<<std::endl;
    TiRuntime runtime = ti_create_runtime(TiArch::TI_ARCH_VULKAN);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(runtime != TI_NULL_HANDLE);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
  }

  if(ti::is_arch_available(TiArch::TI_ARCH_OPENGL))
  {
    std::cout<<"openGL"<<std::endl;
    TiRuntime runitme = ti_create_runtime(TI_ARCH_OPENGL);
    TiError error = ti_get_last_error(0,nullptr);
    std::cout<<error<<std::endl;
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    TI_ASSERT(runtime != TI_NULL_HANDLE);
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
  TI_ASSERT(error == TI_ERROR_ARGUMENT_NULL);
  ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
}

TEST_F (CapiTest, TestBehaviorUnmapMemory)
{
  TiMemoryAllocateInfo* allocate_info = new TiMemoryAllocateInfo;
  allocate_info->size = 1024;
  allocate_info->usage = TI_MEMORY_USAGE_STORAGE_BIT;
  if(ti::is_arch_available(TI_ARCH_VULKAN))
  {
    TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
    TiMemory memory = ti_allocate_memory(runtime,allocate_info);
    ti_map_memory(runtime,memory);
    ti_unmap_memory(runtime,memory);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
  }
}

TEST_F (CapiTest,TestBehaviorAllocateImage)
{
  TiError error = TI_ERROR_SUCCESS;
  TiImageExtent extent;
  extent.height=512;
  extent.width = 512;
  extent.depth = 1;
  extent.array_layer_count = 1;
  TiImageAllocateInfo imageAllocateInfo;
  imageAllocateInfo.dimension=TI_IMAGE_DIMENSION_2D;
  imageAllocateInfo.format = TI_FORMAT_RGBA8;
  imageAllocateInfo.extent = extent;
  imageAllocateInfo.usage = TI_IMAGE_USAGE_STORAGE_BIT;
  imageAllocateInfo.mip_level_count =1;

  if(ti::is_arch_available(TI_ARCH_VULKAN))
  {
    std::cout<<"vulkan"<<std::endl;
    TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
    TiImage image = ti_allocate_image(runtime,&imageAllocateInfo);
    error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error==TI_ERROR_SUCCESS);
    TI_ASSERT(image!=TI_NULL_HANDLE);
    ti_set_last_error(TI_ERROR_SUCCESS,nullptr);

    imageAllocateInfo.dimension = TI_IMAGE_DIMENSION_MAX_ENUM;
    image = ti_allocate_image(runtime,&imageAllocateInfo);
    error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_ARGUMENT_OUT_OF_RANGE);
    TI_ASSERT(image == TI_NULL_HANDLE);
    ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
    imageAllocateInfo.dimension = TI_IMAGE_DIMENSION_2D;

    imageAllocateInfo.format = TI_FORMAT_MAX_ENUM;
    image = ti_allocate_image(runtime,&imageAllocateInfo);
    error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_ARGUMENT_OUT_OF_RANGE);
    TI_ASSERT(image == TI_NULL_HANDLE);
    imageAllocateInfo.format = TI_FORMAT_RGB16F;
  }

  TiImage image = ti_allocate_image(TI_NULL_HANDLE,nullptr);
  TI_ASSERT(image==TI_NULL_HANDLE);
  error = ti_get_last_error(0,nullptr);
  TI_ASSERT(error==TI_ERROR_ARGUMENT_NULL);

  if(ti::is_arch_available(TI_ARCH_OPENGL))
  {
    std::cout<<"openGL"<<std::endl;
    TiRuntime runtime = ti_create_runtime(TI_ARCH_OPENGL);
    TiImage image = ti_allocate_image(runtime, &imageAllocateInfo);          //why not support
    error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    TI_ASSERT(image != TI_NULL_HANDLE);
    ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  }
}

TEST_F(CapiTest,TestBehaviorFreeImage)
{
  TiError error = TI_ERROR_SUCCESS;
  TiImageExtent extent;
  extent.height=512;
  extent.width = 512;
  extent.depth = 1;
  extent.array_layer_count = 1;
  TiImageAllocateInfo imageAllocateInfo;
  imageAllocateInfo.dimension=TI_IMAGE_DIMENSION_2D;
  imageAllocateInfo.format = TI_FORMAT_RGBA8;
  imageAllocateInfo.extent = extent;
  imageAllocateInfo.usage = TI_IMAGE_USAGE_STORAGE_BIT;
  imageAllocateInfo.mip_level_count =1;

  if(ti::is_arch_available(TI_ARCH_VULKAN))
  {
    std::cout<<"vulkan"<<std::endl;
    TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
    TiImage image = ti_allocate_image(runtime,&imageAllocateInfo);
    ti_free_image(runtime,image);
    error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    ti_free_image(TI_NULL_HANDLE,nullptr);
    error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_ARGUMENT_NULL);
    ti_set_last_error(TI_ERROR_SUCCESS,nullptr);

  }
  if(ti::is_arch_available(TI_ARCH_OPENGL))
  {
    std::cout<<"openGL"<<std::endl;
    TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
    TiImage image = ti_allocate_image(runtime,&imageAllocateInfo);
    ti_free_image(runtime,image);
    error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    ti_free_image(TI_NULL_HANDLE,nullptr);
    error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_ARGUMENT_NULL);
    ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
  }
}

TEST_F(CapiTest,TestBehaviorCreateSampler)
{
  TiError error = TI_ERROR_SUCCESS;
  TiSamplerCreateInfo CreateInfo;                            
  CreateInfo.min_filter = TI_FILTER_LINEAR;
  CreateInfo.max_anisotropy = 0.2;
  CreateInfo.address_mode = TI_ADDRESS_MODE_CLAMP_TO_EDGE;
  CreateInfo.mag_filter = TI_FILTER_LINEAR;
  if(ti::is_arch_available(TI_ARCH_VULKAN))
  {
    std::cout<<"vulkan"<<std::endl;
    TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
    TiSampler sampler = ti_create_sampler(runtime,&CreateInfo);
    error = ti_get_last_error(0,nullptr);    
    TI_ASSERT(error == TI_ERROR_NOT_SUPPORTED);
    TI_ASSERT(sampler == TI_NULL_HANDLE);
  }
  if(ti::is_arch_available(TI_ARCH_OPENGL))
  {
    std::cout<<"openGL"<<std::endl;
    TiRuntime runtime = ti_create_runtime(TI_ARCH_OPENGL);
    TiSampler sampler = ti_create_sampler(runtime,&CreateInfo);
    error = ti_get_last_error(0,nullptr);    
    TI_ASSERT(error == TI_ERROR_NOT_SUPPORTED);
    TI_ASSERT(sampler == TI_NULL_HANDLE);
  }
}

TEST_F(CapiTest, TestBehaviorCreateEvent)
{
  if(ti::is_arch_available(TI_ARCH_VULKAN))
  {
    std::cout<<"vulkan"<<std::endl;
    TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
    TiEvent event = ti_create_event(runtime);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    TI_ASSERT(event != TI_NULL_HANDLE);
    event = ti_create_event(TI_NULL_HANDLE);
    error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_ARGUMENT_NULL);
    ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
    error = TI_ERROR_SUCCESS;
  }
  if(ti::is_arch_available(TI_ARCH_OPENGL))
  {
    std::cout<<"openGL"<<std::endl;                                               //openGL不行
    TiRuntime runtime =ti_create_runtime(TI_ARCH_OPENGL);
    TiEvent event = ti_create_event(runtime);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    TI_ASSERT(event != TI_NULL_HANDLE);
  }
}

TEST_F(CapiTest,TestBehaviorDestroyEvent)
{
  if(ti::is_arch_available(TI_ARCH_VULKAN))
  {
    std::cout<<"vulkan"<<std::endl;
    TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
    TiEvent event = ti_create_event(runtime);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    TI_ASSERT(event != TI_NULL_HANDLE);
    ti_destroy_event(event);
    error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error==TI_ERROR_SUCCESS);
    ti_set_last_error(TI_ERROR_SUCCESS,nullptr);
    error = TI_ERROR_SUCCESS;
  }

  if(ti::is_arch_available(TI_ARCH_OPENGL))
  {                                                                                         //openGL不行
    std::cout<<"openGL"<<std::endl;
    TiRuntime runtime =ti_create_runtime(TI_ARCH_OPENGL);
    TiEvent event = ti_create_event(runtime);
    ti_destroy_event(event);
    TiError error = ti_get_last_error(0,nullptr);
    TI_ASSERT(error == TI_ERROR_SUCCESS);
    TI_ASSERT(event != TI_NULL_HANDLE);
  }
}


// TEST_F(CapiTest,TestBehaviorCopyImageDTD)
// {
//   TiError error = TI_ERROR_SUCCESS;
//   TiImageExtent extent;
//   extent.height = 512;
//   extent.width = 512;
//   extent.depth = 1;
//   extent.array_layer_count = 1;
//   TiImageAllocateInfo imageAllocateInfo;
//   imageAllocateInfo.dimension = TI_IMAGE_DIMENSION_2D;
//   imageAllocateInfo.format = TI_FORMAT_RGBA8;
//   imageAllocateInfo.extent = extent;
//   imageAllocateInfo.usage = TI_IMAGE_USAGE_STORAGE_BIT;
//   imageAllocateInfo.mip_level_count = 1;
//   TiImageOffset offset;
//   offset.array_layer_offset = 0;
//   offset.x = 0;
//   offset.y = 0;
//   offset.z = 0;
//   if(ti::is_arch_available(TI_ARCH_VULKAN))
//   {
//     TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
//     TiImage image = ti_allocate_image(runtime, &imageAllocateInfo);
//     TiImageSlice src_image;
//     src_image.image = image;
//     src_image.extent = extent;
//     src_image.mip_level = 0;
//     src_image.offset = offset;
//     TiImageSlice dst_image;
//     ti_copy_image_device_to_device(runtime,&dst_image,&src_image);                                //这个函数用了就编译不过了。
//     error = ti_get_last_error(0,nullptr);
//     TI_ASSERT(error == TI_ERROR_SUCCESS);
//   }
// }

TEST_F(CapiTest,TestBehaviorCopyImageDTD)
{
  TiError error = TI_ERROR_SUCCESS;
  TiMemoryAllocateInfo MemInfo;
  MemInfo.usage = TI_MEMORY_USAGE_STORAGE_BIT;
  MemInfo.size = 1024;
  MemInfo.host_write = TI_TRUE;
  MemInfo.export_sharing = TI_TRUE;
  
  if(ti::is_arch_available(TI_ARCH_VULKAN))
  {
    TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
    TiMemory memory = ti_allocate_memory(runtime,&MemInfo);
    TiMemorySlice dst_memory;
    dst_memory.memory = memory;
    dst_memory.size = 64;
    dst_memory.offset = 128;
    TiMemorySlice src_memory;
    src_memory.memory = memory;
    dst_memory.size = 64;
    dst_memory.offset = 256;

    ti_copy_memory_device_to_device(runtime,&dst_memory,&src_memory);
    error = ti_get_last_error(0,nullptr);



  }
}







