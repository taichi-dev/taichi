#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

TEST_F(CapiTest, TestBehaviorCreateRuntime) {
  auto inner = [this](TiArch arch) {
    TiRuntime runtime = ti_create_runtime(arch);
    TI_ASSERT(runtime == TI_NULL_HANDLE);
    EXPECT_TAICHI_ERROR(TI_ERROR_NOT_SUPPORTED);
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
  EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
}

TEST_F(CapiTest, TestBehaviorGetRuntimeCapabilities) {
  auto inner = [this](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported so the test is skipped", arch);
      return;
    }

    TiRuntime runtime = ti_create_runtime(arch);

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
