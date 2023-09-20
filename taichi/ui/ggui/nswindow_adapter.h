#pragma once

#include "taichi/rhi/common/window_system.h"
#include "taichi/rhi/metal/metal_device.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::metal;

struct NSWindowAdapter {
  void set_content_view(GLFWwindow *glfw_window, metal::MetalSurface *mtl_surf);
};

}  // namespace vulkan

}  // namespace taichi::ui
