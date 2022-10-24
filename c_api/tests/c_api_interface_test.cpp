#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

TEST_F(CapiTest, DryRunRuntime) {
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
    ti::Runtime runtime(arch);
  }
}

TEST_F(CapiTest, DryRunMemoryAllocation) {
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
    ti::NdArray<uint8_t> ndarray = runtime.allocate_ndarray<uint8_t>({100}, {});
  }

  if (capi::utils::is_opengl_available()) {
    // Opengl Runtime
    TiArch arch = TiArch::TI_ARCH_OPENGL;
    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(100);
    ti::NdArray<uint8_t> ndarray = runtime.allocate_ndarray<uint8_t>({100}, {});
  }

  if (capi::utils::is_cuda_available()) {
    // Cuda Runtime
    TiArch arch = TiArch::TI_ARCH_CUDA;
    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(100);
    ti::NdArray<uint8_t> ndarray = runtime.allocate_ndarray<uint8_t>({100}, {});
  }
}

TEST_F(CapiTest, MapDeviceOnlyMemory) {
  {
    ti::Runtime runtime(TI_ARCH_VULKAN);

    runtime.allocate_memory(100);
    char err_msg[256] {0};
    TiError err = ti_get_last_error(sizeof(err_msg), err_msg);

    TI_ASSERT(std::string(err_msg).find("host_read") != std::string::npos);
    TI_ASSERT(std::string(err_msg).find("host_write") != std::string::npos);
    TI_ASSERT(std::string(err_msg).find("host_access") != std::string::npos);
  }
}

TEST_F(CapiTest, DryRunImageAllocation) {
  if (capi::utils::is_vulkan_available()) {
    {
      // Vulkan Runtime
      TiArch arch = TiArch::TI_ARCH_VULKAN;
      ti::Runtime runtime(arch);
      ti::Texture texture =
          runtime.allocate_texture2d(4, 4, TI_FORMAT_RGBA8, TI_NULL_HANDLE);
    }
  }
}

TEST_F(CapiTest, DryRunVulkanAotModule) {
  if (capi::utils::is_vulkan_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    {
      // Vulkan Runtime
      TiArch arch = TiArch::TI_ARCH_VULKAN;
      ti::Runtime runtime(arch);
      ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str());
    }
  }
}

TEST_F(CapiTest, DryRunOpenglAotModule) {
  if (capi::utils::is_opengl_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    {
      // Vulkan Runtime
      TiArch arch = TiArch::TI_ARCH_OPENGL;
      ti::Runtime runtime(arch);

      ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str());
    }
  }
}
