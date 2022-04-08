#include "taichi/ui/common/window_base.h"

TI_UI_NAMESPACE_BEGIN

#define CHECK_WINDOW_SHOWING        \
  TI_ERROR_IF(!config_.show_window, \
              "show_window must be True to use this method")

WindowBase ::WindowBase(AppConfig config) : config_(config) {
  if (config_.show_window) {
    glfw_window_ = create_glfw_window_(config_.name, config_.width,
                                       config_.height, config_.vsync);
    glfwSetWindowUserPointer(glfw_window_, this);
    set_callbacks();
    last_record_time_ = glfwGetTime();
  }
}

void WindowBase::set_callbacks() {
  glfwSetKeyCallback(glfw_window_, key_callback);
  glfwSetCursorPosCallback(glfw_window_, mouse_pos_callback);
  glfwSetMouseButtonCallback(glfw_window_, mouse_button_callback);

  input_handler_.add_key_callback([&](int key, int action) {
    // Catch exception from button_id_to_name().
    try {
      if (action == GLFW_PRESS) {
        events_.push_back({EventType::Press, button_id_to_name(key)});
      } else if (action == GLFW_RELEASE) {
        events_.push_back({EventType::Release, button_id_to_name(key)});
      }
    } catch (const std::runtime_error &e) {
      TI_TRACE("Input: {}.", e.what());
    }
  });
  input_handler_.add_mouse_button_callback([&](int key, int action) {
    // Catch exception from button_id_to_name().
    try {
      if (action == GLFW_PRESS) {
        events_.push_back({EventType::Press, button_id_to_name(key)});
      } else if (action == GLFW_RELEASE) {
        events_.push_back({EventType::Release, button_id_to_name(key)});
      }
    } catch (const std::runtime_error &e) {
      TI_TRACE("Input: {}.", e.what());
    }
  });
}

CanvasBase *WindowBase::get_canvas() {
  return nullptr;
}

void WindowBase::show() {
  CHECK_WINDOW_SHOWING;
  ++frames_since_last_record_;

  double current_time = glfwGetTime();

  if (current_time - last_record_time_ >= 1) {
    double FPS =
        (double)frames_since_last_record_ / (current_time - last_record_time_);
    std::string glfw_window_text =
        config_.name + "  " + std::to_string(FPS) + " FPS";

    glfwSetWindowTitle(glfw_window_, glfw_window_text.c_str());
    last_record_time_ = current_time;
    frames_since_last_record_ = 0;
  }

  glfwPollEvents();
}

bool WindowBase::is_pressed(std::string button) {
  int button_id;
  // Catch exception from buttom_name_to_id().
  try {
    button_id = buttom_name_to_id(button);
  } catch (const std::runtime_error &e) {
    TI_TRACE("Pressed: {}.", e.what());
    return false;
  }
  return input_handler_.is_pressed(button_id) > 0;
}

bool WindowBase::is_running() {
  if (config_.show_window) {
    return !glfwWindowShouldClose(glfw_window_);
  }
  return true;
}

void WindowBase::set_is_running(bool value) {
  if (config_.show_window) {
    glfwSetWindowShouldClose(glfw_window_, !value);
  }
}

std::pair<float, float> WindowBase::get_cursor_pos() {
  CHECK_WINDOW_SHOWING;
  float x = input_handler_.last_x();
  float y = input_handler_.last_y();

  x = x / (float)config_.width;
  y = (config_.height - y) / (float)config_.height;
  return std::make_pair(x, y);
}

std::vector<Event> WindowBase::get_events(EventType tag) {
  CHECK_WINDOW_SHOWING;
  glfwPollEvents();
  std::vector<Event> result;
  std::list<Event>::iterator i = events_.begin();
  while (i != events_.end()) {
    if (i->tag == tag || tag == EventType::Any) {
      result.push_back(*i);
      i = events_.erase(i);
    } else {
      ++i;
    }
  }
  return result;
}

bool WindowBase::get_event(EventType tag) {
  CHECK_WINDOW_SHOWING;
  glfwPollEvents();
  if (events_.size() == 0) {
    return false;
  }
  if (tag == EventType::Any) {
    current_event_ = events_.front();
    events_.pop_front();
    return true;
  } else {
    std::list<Event>::iterator it;
    for (it = events_.begin(); it != events_.end(); ++it) {
      if (it->tag == tag) {
        current_event_ = *it;
        events_.erase(it);
        return true;
      }
    }
    return false;
  }
}

// these 2 are used to export the `current_event` field to python
Event WindowBase::get_current_event() {
  CHECK_WINDOW_SHOWING;
  return current_event_;
}
void WindowBase::set_current_event(const Event &event) {
  CHECK_WINDOW_SHOWING;
  current_event_ = event;
}

WindowBase::~WindowBase() {
  if (config_.show_window) {
    glfwDestroyWindow(glfw_window_);
  }
}

GuiBase *WindowBase::GUI() {
  return nullptr;
}

void WindowBase::key_callback(GLFWwindow *glfw_window,
                              int key,
                              int scancode,
                              int action,
                              int mode) {
  auto window =
      reinterpret_cast<WindowBase *>(glfwGetWindowUserPointer(glfw_window));
  window->input_handler_.key_callback(glfw_window, key, scancode, action, mode);
}

void WindowBase::mouse_pos_callback(GLFWwindow *glfw_window,
                                    double xpos,
                                    double ypos) {
  auto window =
      reinterpret_cast<WindowBase *>(glfwGetWindowUserPointer(glfw_window));
  window->input_handler_.mouse_pos_callback(glfw_window, xpos, ypos);
}

void WindowBase::mouse_button_callback(GLFWwindow *glfw_window,
                                       int button,
                                       int action,
                                       int modifier) {
  auto window =
      reinterpret_cast<WindowBase *>(glfwGetWindowUserPointer(glfw_window));
  window->input_handler_.mouse_button_callback(glfw_window, button, action,
                                               modifier);
}

TI_UI_NAMESPACE_END
