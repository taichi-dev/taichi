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

#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/canvas.h"
#include "taichi/ui/ggui/sceneV2.h"
#include "taichi/ui/ggui/renderer.h"
#include "taichi/ui/common/window_base.h"
#include "taichi/ui/ggui/gui.h"
#include "taichi/ui/ggui/gui_metal.h"
#include "taichi/program/ndarray.h"

namespace taichi::lang {
class Program;
}  // namespace taichi::lang

namespace taichi::ui {

namespace vulkan {

class Window final : public WindowBase {
 public:
  Window(lang::Program *prog, const AppConfig &config);

  void show() override;
  CanvasBase *get_canvas() override;
  SceneBase *get_scene() override;
  GuiBase *gui() override;

  std::pair<uint32_t, uint32_t> get_window_shape() override;

  void write_image(const std::string &filename) override;

  void copy_depth_buffer_to_ndarray(
      const taichi::lang::Ndarray &depth_arr) override;

  std::vector<uint32_t> &get_image_buffer(uint32_t &w, uint32_t &h) override;

  ~Window() override;

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> last_frame_time_;

  std::unique_ptr<Canvas> canvas_;
  std::unique_ptr<SceneV2> scene_;
  std::unique_ptr<GuiBase> gui_;
  std::unique_ptr<Renderer> renderer_;
  bool drawn_frame_{false};
  double fps_limit_{1000.0};
  double limiter_overshoot_{0.0};

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

}  // namespace taichi::ui
