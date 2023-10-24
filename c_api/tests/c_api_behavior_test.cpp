// NOTE: This test intends to fixate error code reporting of all 'unexpected'
// usage of the C-APIs. Normal usages should be covered in other test files like
// `c_api_interface_test.cpp`.
#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "taichi/taichi_cpu.h"
#include "taichi/taichi_cuda.h"
#include "c_api/tests/gtest_fixture.h"

TEST_F(CapiTest, TestBehaviorCreateRuntime) {
  auto inner = [this](TiArch arch) {
    TiRuntime runtime = ti_create_runtime(arch, 0);
    TI_ASSERT(runtime == TI_NULL_HANDLE);
    EXPECT_TAICHI_ERROR(TI_ERROR_NOT_SUPPORTED);
  };

  // Attempt to create runtime for unknown arch.
  inner(TI_ARCH_MAX_ENUM);
}

TEST_F(CapiTest, TestBehaviorDestroyRuntime) {
  // Attempt to destroy null handles.
  ti_destroy_runtime(TI_NULL_HANDLE);
  EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
}

TEST_F(CapiTest, TestBehaviorGetRuntimeCapabilities) {
  auto inner = [this](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported so the test is skipped", arch);
      return;
    }

    ti::Runtime runtime(arch);
    {
      // Two nulls, considerred nop.
      ti_get_runtime_capabilities(runtime, nullptr, nullptr);
      ASSERT_TAICHI_SUCCESS();
    }

    {
      // Count is not null, buffer is null.
      // Usually the case the user look up the number of capabilities will be
      // returned.
      uint32_t capability_count = 0;
      ti_get_runtime_capabilities(runtime, &capability_count, nullptr);
      ASSERT_TAICHI_SUCCESS();
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
      ASSERT_TAICHI_SUCCESS();
      EXPECT_EQ(capabilities.at(0).capability, (TiCapability)0xcbcbcbcb);
      EXPECT_EQ(capabilities.at(0).level, 0xcbcbcbcb);
    }

    {
      // Both non-null.
      // Normal usage.
      uint32_t capability_count = 0;
      ti_get_runtime_capabilities(runtime, &capability_count, nullptr);
      ASSERT_TAICHI_SUCCESS();
      std::vector<TiCapabilityLevelInfo> capabilities(capability_count);
      ti_get_runtime_capabilities(runtime, &capability_count,
                                  capabilities.data());
      ASSERT_TAICHI_SUCCESS();
      for (size_t i = 0; i < capability_count; ++i) {
        TI_ASSERT(capabilities.at(i).capability !=
                  (TiCapability)TI_CAPABILITY_RESERVED);
        TI_ASSERT(capabilities.at(i).level != 0);
      }
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorAllocateMemory) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so this test is skipped", arch);
      return;
    }

    ti::Runtime runtime(arch);

    // Out of memory
    {
      TiMemoryAllocateInfo allocate_info{};
      allocate_info.size = 1000000000000000000;
      allocate_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
      TiMemory memory = ti_allocate_memory(runtime, &allocate_info);
      EXPECT_TAICHI_ERROR(TI_ERROR_OUT_OF_MEMORY);
      TI_ASSERT(memory == TI_NULL_HANDLE);
    }

    // runtime and allocate_info are both null
    {
      TiMemory memory = ti_allocate_memory(TI_NULL_HANDLE, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(memory == TI_NULL_HANDLE);
    }

    // runtime is not null, allocate_info is null
    {
      TiMemory memory = ti_allocate_memory(runtime, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(memory == TI_NULL_HANDLE);
    }

    // runtime is null, allocate is not null;
    {
      TiMemoryAllocateInfo allocate_info{};
      allocate_info.size = 1024;
      TiMemory memory = ti_allocate_memory(TI_NULL_HANDLE, &allocate_info);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(memory == TI_NULL_HANDLE);
    }
  };

  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorFreeMemory) {
  auto inner = [this](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    ti::Runtime runtime(arch);

    // runtime & allocate_info are both null
    {
      ti_free_memory(TI_NULL_HANDLE, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // runtime is null and allocate_info is valid
    {
      TiMemory memory = runtime.allocate_memory(1024);
      ti_free_memory(TI_NULL_HANDLE, memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // runtime is not null and allocate_info is null
    {
      ti_free_memory(runtime, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorMapMemory) {
  auto inner = [this](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(1024);

    // runtime & memory are both null
    {
      void *ptr = ti_map_memory(TI_NULL_HANDLE, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(ptr == nullptr);
    }

    // runtime is null, memory is valid
    {
      void *ptr = ti_map_memory(TI_NULL_HANDLE, memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(ptr == nullptr);
    }

    // runtime is valid, memory is null
    {
      void *ptr = ti_map_memory(runtime, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(ptr == nullptr);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorUnmapMemory) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is nor supported, so the test is skipped", arch);
      return;
    }

    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(1024);

    // runtime & memory are both null
    {
      ti_unmap_memory(TI_NULL_HANDLE, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // runtime is null, memory is valid
    {
      ti_map_memory(TI_NULL_HANDLE, memory);
      ti_unmap_memory(TI_NULL_HANDLE, memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      ti_unmap_memory(runtime, memory);
    }

    // runtime is valid, memory is null
    {
      ti_unmap_memory(runtime, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }
  };
  inner(TI_ARCH_VULKAN);
}

inline TiImageAllocateInfo make_image_allocate_info() {
  TiImageAllocateInfo allocate_info{};
  allocate_info.dimension = TI_IMAGE_DIMENSION_2D;
  allocate_info.format = TI_FORMAT_RGBA8;
  allocate_info.extent.height = 16;
  allocate_info.extent.width = 16;
  allocate_info.extent.depth = 1;
  allocate_info.extent.array_layer_count = 1;
  allocate_info.usage = TI_IMAGE_USAGE_STORAGE_BIT | TI_IMAGE_USAGE_SAMPLED_BIT;
  allocate_info.mip_level_count = 1;
  return allocate_info;
}

TEST_F(CapiTest, TestBehaviorAllocateImage) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    ti::Runtime runtime(arch);

    // Attemp to allocate a 2D image with invalid demension
    {
      TiImageAllocateInfo allocate_info = make_image_allocate_info();
      allocate_info.dimension = TI_IMAGE_DIMENSION_MAX_ENUM;
      TiImage image = ti_allocate_image(runtime, &allocate_info);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_OUT_OF_RANGE);
      TI_ASSERT(image == TI_NULL_HANDLE);
    }

    // Attemp to allocate a 2D image with a invalid format
    {
      TiImageAllocateInfo allocate_info = make_image_allocate_info();
      allocate_info.format = TI_FORMAT_MAX_ENUM;
      TiImage image = ti_allocate_image(runtime, &allocate_info);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_OUT_OF_RANGE);
      TI_ASSERT(image == TI_NULL_HANDLE);
    }

    // runtime & allocate_info are both null
    {
      TiImage image = ti_allocate_image(TI_NULL_HANDLE, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(image == TI_NULL_HANDLE);
    }

    // runtime is null, allocate_info is valid
    {
      TiImageAllocateInfo allocate_info = make_image_allocate_info();
      TiImage image = ti_allocate_image(TI_NULL_HANDLE, &allocate_info);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(image == TI_NULL_HANDLE);
    }

    // runtime is valid, allocate_info is null;
    {
      TiImage image = ti_allocate_image(runtime, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(image == TI_NULL_HANDLE);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorFreeImage) {
  auto inner = [&](TiArch arch) {
    // Attemp to free a normal 2D image
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    ti::Runtime runtime(arch);

    // Runtime & image are both invalid
    {
      ti_free_image(TI_NULL_HANDLE, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // Runtime is null, Image is valid
    {
      TiImage image = runtime.allocate_image(make_image_allocate_info());
      ti_free_image(TI_NULL_HANDLE, image);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // Runtime is valid, image is null
    {
      ti_free_image(runtime, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorCopyMemoryDTD) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    ti::Runtime runtime(arch);
    ti::Memory src = runtime.allocate_memory(2048);
    ti::Memory dst = runtime.allocate_memory(2048);

    {
      src.slice(128, 64).copy_to(dst.slice(1024, 64));
      ASSERT_TAICHI_SUCCESS();
    }

    // Attempt copy memory from the big one to the small one
    {
      src.slice(0, 256).copy_to(dst.slice(0, 64));
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT);
    }

    // runtime is null;
    {
      TiMemorySlice src_memory = src.slice();
      TiMemorySlice dst_memory = dst.slice();
      ti_copy_memory_device_to_device(TI_NULL_HANDLE, &dst_memory, &src_memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // dst memory is null;
    {
      TiMemorySlice src_memory = src.slice();
      TiMemorySlice dst_memory = dst.slice();
      dst_memory.memory = TI_NULL_HANDLE;
      ti_copy_memory_device_to_device(runtime, &dst_memory, &src_memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // src memory is null;
    {
      TiMemorySlice src_memory = src.slice();
      TiMemorySlice dst_memory = dst.slice();
      src_memory.memory = TI_NULL_HANDLE;
      ti_copy_memory_device_to_device(runtime, &dst_memory, &src_memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorLoadAOTModuleVulkan) {
  auto test_behavior_load_aot_module_impl = [this](TiArch arch) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
    const std::string module_path = folder_dir + std::string("/module.tcm");

    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }
    ti::Runtime runtime(arch);

    // AOT module from tcm file, normal usage.
    {
      TiAotModule module = ti_load_aot_module(runtime, module_path.c_str());
      ASSERT_TAICHI_SUCCESS();
      TI_ASSERT(module != TI_NULL_HANDLE);
    }

    // AOT module from filesystem directory, normal usage.
    {
      TiAotModule module = ti_load_aot_module(runtime, folder_dir);
      ASSERT_TAICHI_SUCCESS();
      TI_ASSERT(module != TI_NULL_HANDLE);
    }

    // Attempt to load aot module from tcm file, while runtime is null.
    {
      TiAotModule module =
          ti_load_aot_module(TI_NULL_HANDLE, module_path.c_str());
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(module == TI_NULL_HANDLE);
    }

    // Attempt to load aot module from tcm file, while runtime is null.
    {
      TiAotModule module = ti_load_aot_module(TI_NULL_HANDLE, folder_dir);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(module == TI_NULL_HANDLE);
    }

    // Attempt to load aot module without path.
    {
      TiAotModule module = ti_load_aot_module(runtime, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(module == TI_NULL_HANDLE);
    }
    // Attempt to load aot module with a invalid path.
    {
      TiAotModule module = ti_load_aot_module(runtime, "ssssss///");
      EXPECT_TAICHI_ERROR(TI_ERROR_CORRUPTED_DATA);
      TI_ASSERT(module == TI_NULL_HANDLE);
    }
  };

  test_behavior_load_aot_module_impl(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorDestroyAotModuleVulkan) {
  auto test_behavior_destroy_aot_module_impl = [this](TiArch arch) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
    const std::string module_path = folder_dir + std::string("/module.tcm");
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    // Attempt to destroy a null handle.
    ti_destroy_aot_module(TI_NULL_HANDLE);
    EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
  };

  test_behavior_destroy_aot_module_impl(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorGetCgraphVulkan) {
  auto test_behavior_get_cgraph_impl = [this](TiArch arch) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
    const std::string module_path = folder_dir;
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", arch);
      return;
    }

    ti::Runtime runtime(arch);
    ti::AotModule module = runtime.load_aot_module(module_path.c_str());

    {
      TiComputeGraph cgraph =
          ti_get_aot_module_compute_graph(module, "run_graph");
      ASSERT_TAICHI_SUCCESS();
      TI_ASSERT(cgraph != TI_NULL_HANDLE);
    }

    // Attemp to get compute graph with null module.
    {
      TiComputeGraph cgraph =
          ti_get_aot_module_compute_graph(TI_NULL_HANDLE, "run_graph");
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(cgraph == TI_NULL_HANDLE);
    }

    // Attemp to get compute graph without graph name.
    {
      TiComputeGraph cgraph = ti_get_aot_module_compute_graph(module, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(cgraph == TI_NULL_HANDLE);
    }

    // Attemp to get compute graph with invalid name.
    {
      TiComputeGraph cgraph = ti_get_aot_module_compute_graph(module, "#$#%*(");
      EXPECT_TAICHI_ERROR(TI_ERROR_NAME_NOT_FOUND);
      TI_ASSERT(cgraph == TI_NULL_HANDLE);
    }
  };

  test_behavior_get_cgraph_impl(TI_ARCH_VULKAN);
}
