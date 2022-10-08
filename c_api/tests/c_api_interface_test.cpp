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

TEST(CapiMemory, CrossDeviceMemcpy) {
  {
    uint32_t array_size = 4;

    ti::Runtime cpu_runtime(TiArch::TI_ARCH_X64);
    ti::Runtime vulkan_runtime(TiArch::TI_ARCH_VULKAN);

    ti::NdArray<float> cpu_ndarray =
        cpu_runtime.allocate_ndarray<float>({array_size}, {});
    ti::NdArray<float> vulkan_ndarray =
        vulkan_runtime.allocate_ndarray<float>({array_size}, {});

    const ti::Memory &cpu_mem = cpu_ndarray.memory();
    const ti::Memory &vulkan_mem = vulkan_ndarray.memory();

    TiMemorySlice cpu_mem_slice = {
        .memory = cpu_mem.memory(), .offset = 0, .size = cpu_mem.size()};
    TiMemorySlice vulkan_mem_slice = {
        .memory = vulkan_mem.memory(), .offset = 0, .size = vulkan_mem.size()};

    // CPU to Vulkan
    std::vector<float> cpu_data;
    for (size_t i = 0; i < array_size; i++) {
      cpu_data.push_back(i);
    }
    cpu_mem.write(cpu_data.data(), cpu_data.size() * sizeof(float));
    cpu_runtime.copy_memory_device_to_device(vulkan_mem_slice, cpu_mem_slice);
  }
}
