#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/taichi.hpp"

TEST(CapiDryRun, Runtime) {
  {
    // CPU Runtime
    TiArch arch = TiArch::TI_ARCH_X64;
    ti::Runtime runtime(arch);
  }

  if (capi::utils::is_vulkan_available()) {
    // Vulkan Runtime
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    ti::Runtime runtime(arch);
  }

  if (capi::utils::is_cuda_available()) {
    // Vulkan Runtime
    TiArch arch = TiArch::TI_ARCH_CUDA;
    ti::Runtime runtime(arch);
  }

  if (capi::utils::is_opengl_available()) {
    TiArch arch = TiArch::TI_ARCH_OPENGL;
    TiRuntime runtime = ti_create_runtime(arch);
    ti_destroy_runtime(runtime);
  }
}

TEST(CapiDryRun, MemoryAllocation) {
  {
    // CPU Runtime
    TiArch arch = TiArch::TI_ARCH_X64;
    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(100);
    ti::NdArray<uint8_t> ndarray = runtime.allocate_ndarray<uint8_t>({100}, {});
  }

  if (capi::utils::is_vulkan_available()) {
    // Vulkan Runtime
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(100);
    ti::NdArray<uint8_t> ndarray =
        runtime.allocate_ndarray<uint8_t>({100}, {1});
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
    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(100);
    ti::NdArray<uint8_t> ndarray = runtime.allocate_ndarray<uint8_t>({100}, {});
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
      ti::Runtime runtime(arch);
      ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str().c_str());
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
