#include "nswindow_adapter.h"

#ifdef TI_WITH_METAL

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::metal;

void NSWindowAdapter::set_content_view(GLFWwindow *glfw_window,
                                       metal::MetalSurface *mtl_surf) {

  NSWindow *nswin = glfwGetCocoaWindow(glfw_window);
  nswin.contentView.layer = mtl_surf->mtl_layer();
  nswin.contentView.wantsLayer = YES;
}

} // namespace vulkan

} // namespace taichi::ui

#endif
