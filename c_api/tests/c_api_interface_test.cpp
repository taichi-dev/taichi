#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/taichi_core.h"

TEST(CapiDryRun, Runtime) {
  {
    // CPU Runtime
    TiArch arch = TiArch::TI_ARCH_X64;
    TiRuntime runtime = ti_create_runtime(arch);
    ti_destroy_runtime(runtime);
  }

  if (capi::utils::is_vulkan_available()) {
    // Vulkan Runtime
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    TiRuntime runtime = ti_create_runtime(arch);
    ti_destroy_runtime(runtime);
  }

  if (capi::utils::is_cuda_available()) {
    // Vulkan Runtime
    TiArch arch = TiArch::TI_ARCH_CUDA;
    TiRuntime runtime = ti_create_runtime(arch);
    ti_destroy_runtime(runtime);
  }

  if (capi::utils::is_opengl_available()) {
    TiArch arch = TiArch::TI_ARCH_OPENGL;
    TiRuntime runtime = ti_create_runtime(arch);
    ti_destroy_runtime(runtime);
  }
}

TEST(CapiDryRun, MemoryAllocation) {
  TiMemoryAllocateInfo alloc_info;
  alloc_info.size = 100;
  alloc_info.host_write = false;
  alloc_info.host_read = false;
  alloc_info.export_sharing = false;
  alloc_info.usage = TiMemoryUsageFlagBits::TI_MEMORY_USAGE_STORAGE_BIT;

  {
    // CPU Runtime
    TiArch arch = TiArch::TI_ARCH_X64;
    TiRuntime runtime = ti_create_runtime(arch);

    ti_allocate_memory(runtime, &alloc_info);

    // Unfortunately, memory deallocation for
    // CPU backend has not been implemented yet...

    ti_destroy_runtime(runtime);
  }

  if (capi::utils::is_vulkan_available()) {
    // Vulkan Runtime
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    TiRuntime runtime = ti_create_runtime(arch);

    TiMemory memory = ti_allocate_memory(runtime, &alloc_info);
    ti_free_memory(runtime, memory);

    ti_destroy_runtime(runtime);
  }

  if (capi::utils::is_opengl_available()) {
    // Opengl Runtime
    TiArch arch = TiArch::TI_ARCH_OPENGL;
    TiRuntime runtime = ti_create_runtime(arch);

    TiMemory memory = ti_allocate_memory(runtime, &alloc_info);
    ti_free_memory(runtime, memory);

    ti_destroy_runtime(runtime);
  }

  if (capi::utils::is_cuda_available()) {
    // Cuda Runtime
    TiArch arch = TiArch::TI_ARCH_CUDA;
    TiRuntime runtime = ti_create_runtime(arch);

    TiMemory memory = ti_allocate_memory(runtime, &alloc_info);
    ti_free_memory(runtime, memory);

    ti_destroy_runtime(runtime);
  }
}

TEST(CapiDryRun, VulkanAotModule) {
  if (capi::utils::is_vulkan_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    {
      // Vulkan Runtime
      TiArch arch = TiArch::TI_ARCH_VULKAN;
      TiRuntime runtime = ti_create_runtime(arch);

      TiAotModule aot_mod =
          ti_load_aot_module(runtime, aot_mod_ss.str().c_str());
      ti_destroy_aot_module(aot_mod);

      ti_destroy_runtime(runtime);
    }
  }
}

TEST(CapiDryRun, OpenglAotModule) {
  if (capi::utils::is_opengl_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    {
      // Vulkan Runtime
      TiArch arch = TiArch::TI_ARCH_OPENGL;
      TiRuntime runtime = ti_create_runtime(arch);

      TiAotModule aot_mod =
          ti_load_aot_module(runtime, aot_mod_ss.str().c_str());
      ti_destroy_aot_module(aot_mod);

      ti_destroy_runtime(runtime);
    }
  }
}
