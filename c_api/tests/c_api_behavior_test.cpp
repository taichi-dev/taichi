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
    ti_destroy_runtime(runtime);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
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
    ti_destroy_runtime(runtime);
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorAllocateMemory) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so this test is skipped", arch);
      return;
    }
    // Attempt to allocate memory with size of 1024
    TiRuntime runtime = ti_create_runtime(arch);
    TiMemoryAllocateInfo allocateInfo;
    allocateInfo.size = 1024;
    allocateInfo.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    TiMemory memory = ti_allocate_memory(runtime, &allocateInfo);
    TI_ASSERT(memory != TI_NULL_HANDLE);
    ti_free_memory(runtime, memory);
    ti_destroy_runtime(runtime);
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorAllocInvalidMemory) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so this test is skipped", arch);
      return;
    }
    // Attemp to run out of memory
    TiRuntime runtime = ti_create_runtime(arch);
    TiMemoryAllocateInfo allocateInfo;
    allocateInfo.size = 1000000000000000000;
    allocateInfo.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    TiMemory memory = ti_allocate_memory(runtime, &allocateInfo);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_OUT_OF_MEMORY);
    TI_ASSERT(memory == TI_NULL_HANDLE);
    ti_destroy_runtime(runtime);
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorAllocMemoryNoArg) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so this test is skipped", arch);
      return;
    }
    // runtime and allocate_info are both null
    ti_allocate_memory(TI_NULL_HANDLE, nullptr);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorAllocMemoryNoAllocInfo) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so this test is skipped", arch);
      return;
    }
    // runtime is not null, allocate_info is null
    TiRuntime runtime = ti_create_runtime(arch);
    ti_allocate_memory(runtime, nullptr);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    ti_destroy_runtime(runtime);
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorAllocMemoryNoRuntime) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so this test is skipped", arch);
      return;
    }
    // runtime is null, allocate is not null;
    TiMemoryAllocateInfo allocateInfo;
    allocateInfo.size = 1024;
    ti_allocate_memory(TI_NULL_HANDLE, &allocateInfo);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorFreeMemory) {
  auto inner = [](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    {
      TiRuntime runtime = ti_create_runtime(arch);
      TiMemoryAllocateInfo *allocateInfo = new TiMemoryAllocateInfo;
      allocateInfo->size = 1024;
      allocateInfo->usage = TI_MEMORY_USAGE_STORAGE_BIT;
      TiMemory memory = ti_allocate_memory(runtime, allocateInfo);
      ti_free_memory(runtime, memory);
      CHECK_TAICHI_SUCCESS();
      ti_destroy_runtime(runtime);
    }

    // runtime & allocate_info are both null
    {
      ti_free_memory(TI_NULL_HANDLE, nullptr);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    }

    // runtime is null and allocate_info is valid
    {
      TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
      TiMemoryAllocateInfo allocateInfo;
      allocateInfo.size = 1024;
      allocateInfo.usage = TI_MEMORY_USAGE_STORAGE_BIT;
      TiMemory memory = ti_allocate_memory(runtime, &allocateInfo);
      ti_free_memory(TI_NULL_HANDLE, memory);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
      ti_destroy_runtime(runtime);
    }

    // runtime is not null and allocate_info is null
    {
      TiRuntime runtime = ti_create_runtime(arch);
      ti_free_memory(runtime, nullptr);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorMapMemory) {
  TiMemoryAllocateInfo allocate_info;
  allocate_info.size = 1024;
  allocate_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;

  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    {
      TiRuntime runtime = ti_create_runtime(arch);
      TiMemory memory = ti_allocate_memory(runtime, &allocate_info);
      TI_ASSERT(memory != TI_NULL_HANDLE);
      CHECK_TAICHI_SUCCESS();
      ti_map_memory(runtime, memory);
      CHECK_TAICHI_SUCCESS();
      ti_unmap_memory(runtime, memory);
      ti_destroy_runtime(runtime);
    }

    // runtime & memory are both null
    {
      TiRuntime runtime = ti_create_runtime(arch);
      ti_map_memory(TI_NULL_HANDLE, TI_NULL_HANDLE);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
      ti_destroy_runtime(runtime);
    }

    // runtime is null, memory is valid
    {
      TiRuntime runtime = ti_create_runtime(arch);
      TiMemory memory = ti_allocate_memory(runtime, &allocate_info);
      ti_map_memory(TI_NULL_HANDLE, memory);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
      ti_unmap_memory(runtime, memory);
      ti_destroy_runtime(runtime);
    }

    // runtime is valid, memory is null
    {
      TiRuntime runtime = ti_create_runtime(arch);
      ti_map_memory(runtime, TI_NULL_HANDLE);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
      ti_destroy_runtime(runtime);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorUnmapMemory) {
  TiMemoryAllocateInfo allocate_info;
  allocate_info.size = 1024;
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is nor supported, so the test is skipped", arch);
      return;
    }

    {
      TiRuntime runtime = ti_create_runtime(arch);
      allocate_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
      TiMemory memory = ti_allocate_memory(runtime, &allocate_info);
      ti_map_memory(runtime, memory);
      ti_unmap_memory(runtime, memory);
      CHECK_TAICHI_SUCCESS();
      ti_destroy_runtime(runtime);
    }

    // runtime & memory are both null
    {
      ti_unmap_memory(TI_NULL_HANDLE, TI_NULL_HANDLE);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    }

    // runtime is null, memory is valid
    {
      TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
      TiMemory memory = ti_allocate_memory(runtime, &allocate_info);
      ti_unmap_memory(TI_NULL_HANDLE, memory);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
      ti_free_memory(runtime, memory);
      ti_destroy_runtime(runtime);
    }

    // runtime is valid, memory is valid
    {
      TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN);
      ti_unmap_memory(runtime, TI_NULL_HANDLE);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
      ti_destroy_runtime(runtime);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TiImageAllocateInfo get_image_allocate_info() {
  TiImageExtent extent;
  extent.height = 512;
  extent.width = 512;
  extent.depth = 1;
  extent.array_layer_count = 1;
  TiImageAllocateInfo imageAllocateInfo;
  imageAllocateInfo.dimension = TI_IMAGE_DIMENSION_2D;
  imageAllocateInfo.format = TI_FORMAT_RGBA8;
  imageAllocateInfo.extent = extent;
  imageAllocateInfo.usage = TI_IMAGE_USAGE_STORAGE_BIT;
  imageAllocateInfo.mip_level_count = 1;
  return imageAllocateInfo;
}

TEST_F(CapiTest, TestBehaviorAllocateImage) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    // Attemp to allocate a normal 2D image
    {
      TiImageExtent extent;
      extent.height = 512;
      extent.width = 512;
      extent.depth = 1;
      extent.array_layer_count = 1;
      TiImageAllocateInfo imageAllocateInfo;
      imageAllocateInfo.dimension = TI_IMAGE_DIMENSION_2D;
      imageAllocateInfo.format = TI_FORMAT_RGBA8;
      imageAllocateInfo.extent = extent;
      imageAllocateInfo.usage = TI_IMAGE_USAGE_STORAGE_BIT;
      imageAllocateInfo.mip_level_count = 1;
      TiRuntime runtime = ti_create_runtime(arch);
      TiImage image = ti_allocate_image(runtime, &imageAllocateInfo);
      CHECK_TAICHI_SUCCESS();
      TI_ASSERT(image != TI_NULL_HANDLE);

      imageAllocateInfo.usage = TI_IMAGE_USAGE_SAMPLED_BIT;
      image = ti_allocate_image(runtime, &imageAllocateInfo);
      CHECK_TAICHI_SUCCESS();
      TI_ASSERT(image != TI_NULL_HANDLE);
      imageAllocateInfo.usage = TI_IMAGE_USAGE_STORAGE_BIT;

      ti_free_image(runtime, image);
      ti_destroy_runtime(runtime);
    }

    // Attemp to allocate a 2D image with invalid demension
    {
      TiRuntime runtime = ti_create_runtime(arch);
      auto imageAllocateInfo = get_image_allocate_info();
      imageAllocateInfo.dimension = TI_IMAGE_DIMENSION_MAX_ENUM;
      TiImage image = ti_allocate_image(runtime, &imageAllocateInfo);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_OUT_OF_RANGE);
      TI_ASSERT(image == TI_NULL_HANDLE);
      ti_free_image(runtime, image);
      ti_destroy_runtime(runtime);
    }

    // Attemp to allocate a 2D image with a invalid format
    {
      TiRuntime runtime = ti_create_runtime(arch);
      auto imageAllocateInfo = get_image_allocate_info();
      imageAllocateInfo.format = TI_FORMAT_MAX_ENUM;
      TiImage image = ti_allocate_image(runtime, &imageAllocateInfo);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_OUT_OF_RANGE);
      imageAllocateInfo.format = TI_FORMAT_BGRA8;
      ti_free_image(runtime, image);
      ti_destroy_runtime(runtime);
    }

    // runtime & imageAllocateInfo are both null
    {
      TiRuntime runtime = ti_create_runtime(arch);
      auto image = ti_allocate_image(TI_NULL_HANDLE, TI_NULL_HANDLE);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
      ti_free_image(runtime, image);
      ti_destroy_runtime(runtime);
    }

    // runtime is null, imageAllocateInfo is valid
    {
      TiRuntime runtime = ti_create_runtime(arch);
      TiImageAllocateInfo imageAllocateInfo = get_image_allocate_info();
      TiImage image = ti_allocate_image(TI_NULL_HANDLE, &imageAllocateInfo);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
      ti_free_image(runtime, image);
      ti_destroy_runtime(runtime);
    }

    // runtime is valid, imageAllocateInfo is null;
    {
      TiRuntime runtime = ti_create_runtime(arch);
      TiImage image = ti_allocate_image(runtime, TI_NULL_HANDLE);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
      ti_free_image(runtime, image);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
      ti_destroy_runtime(runtime);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorFreeImage) {
  TiImageExtent extent;
  extent.height = 512;
  extent.width = 512;
  extent.depth = 1;
  extent.array_layer_count = 1;
  TiImageAllocateInfo imageAllocateInfo;
  imageAllocateInfo.dimension = TI_IMAGE_DIMENSION_2D;
  imageAllocateInfo.format = TI_FORMAT_RGBA8;
  imageAllocateInfo.extent = extent;
  imageAllocateInfo.usage = TI_IMAGE_USAGE_STORAGE_BIT;
  imageAllocateInfo.mip_level_count = 1;

  auto inner = [&](TiArch arch) {
    // Attemp to free a normal 2D image
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    TiRuntime runtime = ti_create_runtime(arch);
    TiImage image = ti_allocate_image(runtime, &imageAllocateInfo);
    ti_free_image(runtime, image);
    CHECK_TAICHI_SUCCESS();

    // Runtime & image are both invalid
    {
      ti_free_image(TI_NULL_HANDLE, TI_NULL_HANDLE);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    }

    // Runtime is null, Image is valid
    {
      ti_free_image(TI_NULL_HANDLE, image);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    }

    // Runtime is valid, image is null
    {
      ti_free_image(runtime, TI_NULL_HANDLE);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    }

    ti_destroy_runtime(runtime);
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorCreateEvent) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }
    TiRuntime runtime = ti_create_runtime(arch);
    TiEvent event = ti_create_event(runtime);
    CHECK_TAICHI_SUCCESS();

    // Runtime is null
    {
      event = ti_create_event(TI_NULL_HANDLE);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    }
    ti_destroy_runtime(runtime);
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorDestroyEvent) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    TiRuntime runtime = ti_create_runtime(arch);
    TiEvent event = ti_create_event(runtime);
    ti_destroy_event(event);
    CHECK_TAICHI_SUCCESS();

    // Attemp to destroy a null event
    {
      ti_destroy_event(TI_NULL_HANDLE);
      CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    }
    ti_destroy_runtime(runtime);
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorCopyMemoryDTD) {
  TiMemoryAllocateInfo MallocateInfo;
  MallocateInfo.size = 2048;
  MallocateInfo.usage = TI_MEMORY_USAGE_STORAGE_BIT;
  MallocateInfo.export_sharing = TI_TRUE;
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }
    TiRuntime runtime = ti_create_runtime(arch);
    TiMemory memory = ti_allocate_memory(runtime, &MallocateInfo);
    TiMemorySlice src_memory;
    src_memory.memory = memory;
    src_memory.offset = 128;
    src_memory.size = 64;
    TiMemorySlice dst_memory;
    dst_memory.memory = memory;
    dst_memory.offset = 1024;
    dst_memory.size = 64;
    ti_copy_memory_device_to_device(runtime, &dst_memory, &src_memory);
    CHECK_TAICHI_SUCCESS();

    // Attempt copy memory from the big one to the small one
    src_memory.size = 256;
    src_memory.offset = 512;
    dst_memory.offset = 1152;
    ti_copy_memory_device_to_device(runtime, &dst_memory, &src_memory);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_INVALID_ARGUMENT);
    dst_memory.size = 64;

    // runtime is null;
    ti_copy_memory_device_to_device(TI_NULL_HANDLE, &dst_memory, &src_memory);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);

    // dst memory is null;
    dst_memory.memory = TI_NULL_HANDLE;
    ti_copy_memory_device_to_device(runtime, &dst_memory, &src_memory);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);

    // src memory is null;
    src_memory.memory = TI_NULL_HANDLE;
    ti_copy_memory_device_to_device(runtime, &dst_memory, &src_memory);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);

    ti_free_memory(runtime, memory);
    ti_destroy_runtime(runtime);
  };
  inner(TI_ARCH_VULKAN);
}

void test_behavior_load_aot_module_impl(TiArch arch) {
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  const std::string module_path = folder_dir + std::string("/module.tcm");

  if (!ti::is_arch_available(arch)) {
    TI_WARN("arch {} is not supported, so the test is skipped", arch);
    return;
  }
  TiRuntime runtime = ti_create_runtime(arch);

  // AOT module from tcm file, normal usage.
  {
    TiAotModule module = ti_load_aot_module(runtime, module_path.c_str());
    CHECK_TAICHI_SUCCESS();
    TI_ASSERT(module != TI_NULL_HANDLE);
  }

  // AOT module from filesystem directory, normal usage.
  {
    TiAotModule module = ti_load_aot_module(runtime, folder_dir);
    CHECK_TAICHI_SUCCESS();
    TI_ASSERT(module != TI_NULL_HANDLE);
  }

  // Attempt to load aot module from tcm file, while runtime is null.
  {
    TiAotModule module =
        ti_load_aot_module(TI_NULL_HANDLE, module_path.c_str());
    CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    TI_ASSERT(module == TI_NULL_HANDLE);
  }

  // Attempt to load aot module from tcm file, while runtime is null.
  {
    TiAotModule module = ti_load_aot_module(TI_NULL_HANDLE, folder_dir);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    TI_ASSERT(module == TI_NULL_HANDLE);
  }

  // Attempt to load aot module without path.
  {
    TiAotModule module = ti_load_aot_module(runtime, nullptr);
    CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
    TI_ASSERT(module == TI_NULL_HANDLE);
  }
  // Attempt to load aot module with a invalid path.
  {
    TiAotModule module = ti_load_aot_module(runtime, "ssssss///");
    CHECK_TAICHI_ERROR_IS(TI_ERROR_CORRUPTED_DATA);
    TI_ASSERT(module == TI_NULL_HANDLE);
  }
  ti_destroy_runtime(runtime);
}

TEST_F(CapiTest, TestBehaviorLoadAOTModuleVulkan) {
  test_behavior_load_aot_module_impl(TI_ARCH_VULKAN);
}

void test_behavior_destroy_aot_module_impl(TiArch arch) {
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  const std::string module_path = folder_dir + std::string("/module.tcm");
  if (!ti::is_arch_available(arch)) {
    TI_WARN("arch {} is not supported, so the test is skipped", arch);
    return;
  }

  TiRuntime runtime = ti_create_runtime(arch);
  TiAotModule module = ti_load_aot_module(runtime, module_path.c_str());
  TI_ASSERT(module != TI_NULL_HANDLE);
  ti_destroy_aot_module(module);
  CHECK_TAICHI_SUCCESS();
  ti_destroy_runtime(runtime);

  // Attempt to destroy a null handle.
  ti_destroy_aot_module(TI_NULL_HANDLE);
  CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
}

TEST_F(CapiTest, TestBehaviorDestroyAotModuleVulkan) {
  test_behavior_destroy_aot_module_impl(TI_ARCH_VULKAN);
}

void test_behavior_get_cgraph_impl(TiArch arch) {
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  const std::string module_path = folder_dir;
  if (!ti::is_arch_available(arch)) {
    TI_WARN("arch {} is not supported, so the test is skipped", arch);
    return;
  }

  TiRuntime runtime = ti_create_runtime(arch);
  TiAotModule module = ti_load_aot_module(runtime, module_path.c_str());
  TiComputeGraph cgraph = ti_get_aot_module_compute_graph(module, "run_graph");
  CHECK_TAICHI_SUCCESS();
  TI_ASSERT(cgraph != TI_NULL_HANDLE);

  // Attemp to get compute graph with null module.
  cgraph = ti_get_aot_module_compute_graph(TI_NULL_HANDLE, "run_graph");
  CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
  TI_ASSERT(cgraph == TI_NULL_HANDLE);

  // Attemp to get compute graph without graph name.
  cgraph = ti_get_aot_module_compute_graph(module, nullptr);
  CHECK_TAICHI_ERROR_IS(TI_ERROR_ARGUMENT_NULL);
  TI_ASSERT(cgraph == TI_NULL_HANDLE);

  // Attemp to get compute graph with invalid name.
  cgraph = ti_get_aot_module_compute_graph(module, "#$#%*(");
  CHECK_TAICHI_ERROR_IS(TI_ERROR_NAME_NOT_FOUND);
  TI_ASSERT(cgraph == TI_NULL_HANDLE);

  ti_destroy_runtime(runtime);
}

TEST_F(CapiTest, TestBehaviorGetCgraphVulkan) {
  test_behavior_get_cgraph_impl(TI_ARCH_VULKAN);
}
