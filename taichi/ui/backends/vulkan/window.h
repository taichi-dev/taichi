#pragma once

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <optional>
#include <set>
#include "taichi/ui/utils/utils.h"
#include <memory>

#include "taichi/ui/backends/vulkan/swap_chain.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/canvas.h"
#include "taichi/ui/backends/vulkan/renderer.h"
#include "taichi/ui/common/window_base.h"
#include "taichi/ui/backends/vulkan/gui.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class Window final : public WindowBase {
 public:
  Window(AppConfig config);

  virtual void show() override;
  virtual CanvasBase *get_canvas() override;
  virtual GuiBase *GUI() override;

  ~Window();

 private:
  std::unique_ptr<Canvas> canvas_;
  AppContext app_context_;
  Gui gui_;
  std::unique_ptr<Renderer> renderer_;

 private:
  void init();

  void init_window();

  void init_vulkan();

  void prepare_for_next_frame();

  void draw_frame();

  void present_frame();

  void update_image_index();

  void cleanup_swap_chain();

  void cleanup();

  void recreate_swap_chain();

  static void framebuffer_resize_callback(GLFWwindow *glfw_window_,
                                          int width,
                                          int height);
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
