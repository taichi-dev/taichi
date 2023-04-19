#ifdef TI_WITH_VULKAN
// NOTE: This must be included before `GLFW/glfw3.h` is included
#include "taichi/rhi/vulkan/vulkan_common.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif

#include "taichi/rhi/common/window_system.h"
#include "taichi/rhi/impl_support.h"

#include <mutex>
#include <array>
#include <iostream>

namespace taichi::lang::window_system {

#ifdef TI_WITH_GLFW
struct GLFWState {
  std::mutex mutex;
  int glfw_ref_count = 0;
};

static GLFWState glfw_state;

static void glfw_error_callback(int code, const char *description) {
  std::array<char, 1024> buf;
  snprintf(buf.data(), buf.size(), "GLFW Error %d: %s", code, description);
  RHI_LOG_ERROR(buf.data());
}

bool glfw_context_acquire() {
  std::lock_guard lg(glfw_state.mutex);
  if (glfw_state.glfw_ref_count == 0) {
#ifdef TI_WITH_VULKAN
    // This must be done before glfwInit() on macOS
    // Ref: https://github.com/taichi-dev/taichi/pull/4813
    // Here the `vkGetInstanceProcAddr` comes from Volk
    if (vulkan::is_vulkan_api_available()) {
      glfwInitVulkanLoader(vkGetInstanceProcAddr);
    }
#endif

    auto res = glfwInit();
    if (res != GLFW_TRUE) {
      return false;
    }

    glfwSetErrorCallback(glfw_error_callback);
  }
  glfw_state.glfw_ref_count++;
  return true;
}

void glfw_context_release() {
  std::lock_guard lg(glfw_state.mutex);
  if (glfw_state.glfw_ref_count <= 0) {
    assert(false && "GLFW ref count underflow");
  }
  glfw_state.glfw_ref_count--;
  if (glfw_state.glfw_ref_count == 0) {
    glfwTerminate();
  } else if (glfw_state.glfw_ref_count < 0) {
    assert(false && "GLFW context double release?");
  }
}

#else

bool glfw_context_acquire() {
  return false;
}

void glfw_context_release() {
  return;
}

#endif  // TI_WITH_GLFW

}  // namespace taichi::lang::window_system
