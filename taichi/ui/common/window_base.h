#pragma once

#include <string>
#include "taichi/ui/utils/utils.h"
#include "input_handler.h"

#include <vector>
#include <unordered_map>
#include <queue>
#include <list>
#include <tuple>

#include "taichi/ui/common/canvas_base.h"
#include "taichi/ui/common/event.h"
#include "taichi/ui/common/gui_base.h"
#include "taichi/ui/common/app_config.h"
#include "taichi/program/ndarray.h"

struct GLFWwindow;

namespace taichi::ui {

class WindowBase {
 public:
  bool is_pressed(std::string button);

  bool is_running();

  void set_is_running(bool value);

  std::pair<float, float> get_cursor_pos();

  std::vector<Event> get_events(EventType tag);

  bool get_event(EventType tag);

  Event get_current_event();

  void set_current_event(const Event &event);

  virtual CanvasBase *get_canvas();

  virtual SceneBase *get_scene();

  virtual void show();

  virtual std::pair<uint32_t, uint32_t> get_window_shape() = 0;

  virtual void write_image(const std::string &filename) = 0;

  virtual void copy_depth_buffer_to_ndarray(const taichi::lang::Ndarray &) = 0;

  virtual std::vector<uint32_t> &get_image_buffer(uint32_t &w, uint32_t &h) = 0;

  virtual GuiBase *gui();

  virtual ~WindowBase();

 protected:
  AppConfig config_;
  GLFWwindow *glfw_window_{nullptr};
  InputHandler input_handler_;

  // used for FPS counting
  double last_record_time_{0.0};
  int frames_since_last_record_{0};

  std::list<Event> events_;
  Event current_event_{EventType::Any, ""};

 protected:
  explicit WindowBase(AppConfig config);

  void set_callbacks();

  static void key_callback(GLFWwindow *glfw_window,
                           int key,
                           int scancode,
                           int action,
                           int mode);

  static void mouse_pos_callback(GLFWwindow *glfw_window,
                                 double xpos,
                                 double ypos);

  static void mouse_button_callback(GLFWwindow *glfw_window,
                                    int button,
                                    int action,
                                    int modifier);
};

}  // namespace taichi::ui
