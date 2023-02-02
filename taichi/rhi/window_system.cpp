#include "window_system.h"
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
