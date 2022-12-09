#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

TEST_F(CapiTest, DryRunAvailableArchs) {
  std::vector<TiArch> archs = ti::get_available_archs();
}

TEST_F(CapiTest, DryRunRuntime) {
  {
    // CPU Runtime
    TiArch arch = TiArch::TI_ARCH_X64;
    ti::Runtime runtime(arch);
    runtime.destroy();
  }

  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    // Vulkan Runtime
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    ti::Runtime runtime(arch);
    runtime.destroy();
  }

  if (ti::is_arch_available(TI_ARCH_CUDA)) {
    // CUDA Runtime
    TiArch arch = TiArch::TI_ARCH_CUDA;
    ti::Runtime runtime(arch);
    runtime.destroy();
  }

  if (ti::is_arch_available(TI_ARCH_OPENGL)) {
    // openGL Runtime
    TiArch arch = TiArch::TI_ARCH_OPENGL;
    ti::Runtime runtime(arch);
    runtime.destroy();
  }
}

TEST_F(CapiTest, DryRunCapabilities) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    // Vulkan Runtime
    {
      ti::Runtime runtime(TI_ARCH_VULKAN);
      auto devcaps = runtime.get_capabilities();
      auto level = devcaps.get(TI_CAPABILITY_SPIRV_VERSION);
      assert(level >= 0x10000);
    }
  }
}

TEST_F(CapiTest, SetCapabilities) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    // Vulkan Runtime
    {
      ti::Runtime runtime(TI_ARCH_VULKAN);

      {
        auto devcaps = ti::CapabilityLevelConfig::builder()
                           .spirv_version(1, 3)
                           .spirv_has_atomic_float64_add()
                           .build();
        runtime.set_capabilities_ext(devcaps);
        auto devcaps2 = runtime.get_capabilities();
        TI_ASSERT(devcaps2.get(TI_CAPABILITY_SPIRV_VERSION) == 0x10300);
        TI_ASSERT(devcaps2.get(TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64_ADD) ==
                  1);
        TI_ASSERT(devcaps2.get(TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64) == 0);
      }
      {
        auto devcaps =
            ti::CapabilityLevelConfig::builder().spirv_version(1, 4).build();
        runtime.set_capabilities_ext(devcaps);
        auto devcaps2 = runtime.get_capabilities();
        TI_ASSERT(devcaps2.get(TI_CAPABILITY_SPIRV_VERSION) == 0x10400);
        TI_ASSERT(devcaps2.get(TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64_ADD) ==
                  0);
        TI_ASSERT(devcaps2.get(TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64) == 0);
      }
      {
        auto devcaps = ti::CapabilityLevelConfig::builder()
                           .spirv_version(1, 5)
                           .spirv_has_atomic_float64()
                           .spirv_has_atomic_float64(false)
                           .spirv_has_atomic_float64(true)
                           .build();
        runtime.set_capabilities_ext(devcaps);
        auto devcaps2 = runtime.get_capabilities();
        TI_ASSERT(devcaps2.get(TI_CAPABILITY_SPIRV_VERSION) == 0x10500);
        TI_ASSERT(devcaps2.get(TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64_ADD) ==
                  0);
        TI_ASSERT(devcaps2.get(TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64) == 1);
      }
    }
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

  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    // Vulkan Runtime
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(100);
    ti::NdArray<uint8_t> ndarray = runtime.allocate_ndarray<uint8_t>({100}, {});
  }

  if (ti::is_arch_available(TI_ARCH_OPENGL)) {
    // Opengl Runtime
    TiArch arch = TiArch::TI_ARCH_OPENGL;
    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(100);
    ti::NdArray<uint8_t> ndarray = runtime.allocate_ndarray<uint8_t>({100}, {});
  }

  if (ti::is_arch_available(TI_ARCH_CUDA)) {
    // Cuda Runtime
    TiArch arch = TiArch::TI_ARCH_CUDA;
    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(100);
    ti::NdArray<uint8_t> ndarray = runtime.allocate_ndarray<uint8_t>({100}, {});
  }
}

TEST_F(CapiTest, FailMapDeviceOnlyMemory) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    ti::Runtime runtime(TI_ARCH_VULKAN);

    ti::Memory mem = runtime.allocate_memory(100);
    mem.map();

    char err_msg[1024]{0};
    TiError err = ti_get_last_error(sizeof(err_msg), err_msg);

    TI_ASSERT(err == TI_ERROR_INVALID_STATE);
    TI_ASSERT(std::string(err_msg).find("Assertion failure") !=
              std::string::npos);
    TI_ASSERT(std::string(err_msg).find("map") != std::string::npos);
    TI_ASSERT(std::string(err_msg).find("RhiResults::success") !=
              std::string::npos);

    ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  }
}

TEST_F(CapiTest, FailOutOfRangeReadWrite) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    ti::Runtime runtime(TI_ARCH_VULKAN);

    std::vector<float> data(101);
    ti::NdArray<float> arr = runtime.allocate_ndarray<float>({50}, {2});

    TI_ASSERT(arr.elem_count() == 50);
    TI_ASSERT(arr.scalar_count() == 100);

    arr.write(data);

    char err_msg[1024]{0};
    TiError err = ti_get_last_error(sizeof(err_msg), err_msg);

    TI_ASSERT(err == TI_ERROR_ARGUMENT_OUT_OF_RANGE);

    ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  }
}

TEST_F(CapiTest, DryRunImageAllocation) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
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
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
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
  if (ti::is_arch_available(TI_ARCH_OPENGL)) {
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

TEST_F(CapiTest, TestLoadTcmAotModule) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir << "/module.tcm";

    {
      // Vulkan Runtime
      TiArch arch = TiArch::TI_ARCH_VULKAN;
      ti::Runtime runtime(arch);
      ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str());
      ti::Kernel run = aot_mod.get_kernel("run");
      ti::NdArray<int32_t> arr =
          runtime.allocate_ndarray<int32_t>({16}, {}, true);
      run[0] = arr;
      run.launch();
      runtime.wait();
      std::vector<int32_t> data(16);
      arr.read(data);
      for (int32_t i = 0; i < 16; ++i) {
        TI_ASSERT(data.at(i) == i);
      }
    }
  }
}

TEST_F(CapiTest, TestCreateTcmAotModule) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir << "/module.tcm";

    std::vector<uint8_t> tcm;
    {
      std::fstream f(aot_mod_ss.str(),
                     std::ios::in | std::ios::binary | std::ios::ate);
      TI_ASSERT(f.is_open());
      tcm.resize(f.tellg());
      f.seekg(std::ios::beg);
      f.read((char *)tcm.data(), tcm.size());
    }

    {
      // Vulkan Runtime
      TiArch arch = TiArch::TI_ARCH_VULKAN;
      ti::Runtime runtime(arch);
      ti::AotModule aot_mod = runtime.create_aot_module(tcm);
      ti::Kernel run = aot_mod.get_kernel("run");
      ti::NdArray<int32_t> arr =
          runtime.allocate_ndarray<int32_t>({16}, {}, true);
      run[0] = arr;
      run.launch();
      runtime.wait();
      std::vector<int32_t> data(16);
      arr.read(data);
      for (int32_t i = 0; i < 16; ++i) {
        TI_ASSERT(data.at(i) == i);
      }
    }
  }
}
