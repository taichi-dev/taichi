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

namespace taichi {
namespace lang {
class Program;
}  // namespace lang
}  // namespace taichi

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class Window final : public WindowBase {
 public:
  Window(lang::Program *prog, const AppConfig &config);

  virtual void show() override;
  virtual CanvasBase *get_canvas() override;
  virtual GuiBase *GUI() override;

  void write_image(const std::string &filename) override;

  ~Window();

 private:
  std::unique_ptr<Canvas> canvas_;
  std::unique_ptr<Gui> gui_;
  std::unique_ptr<Renderer> renderer_;
  bool drawn_frame_{false};

 private:
  void init(lang::Program *prog, const AppConfig &config);

  void prepare_for_next_frame();

  void draw_frame();

  void present_frame();

  void resize();

  static void framebuffer_resize_callback(GLFWwindow *glfw_window_,
                                          int width,
                                          int height);
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
