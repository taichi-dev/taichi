#pragma once
#include <memory>
#include <functional>
#include <vector>
#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

class InputHandler {
 public:
  void key_callback(GLFWwindow *window,
                    int key,
                    int scancode,
                    int action,
                    int mode) {
    if (action == GLFW_PRESS) {
      keys_[key] = true;
    } else if (action == GLFW_RELEASE) {
      keys_[key] = false;
    }
    for (auto f : user_key_callbacks_) {
      f(key, action);
    }
  }

  void mouse_pos_callback(GLFWwindow *window, double xpos, double ypos) {
    if (first_mouse_) {
      last_x_ = xpos;
      last_y_ = ypos;
      first_mouse_ = false;
    }

    last_x_ = xpos;
    last_y_ = ypos;

    for (auto f : user_mouse_pos_callbacks_) {
      f(xpos, ypos);
    }
  }

  void mouse_button_callback(GLFWwindow *window,
                             int button,
                             int action,
                             int modifier) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      if (action == GLFW_PRESS) {
        left_mouse_down_ = true;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
      }
      if (action == GLFW_RELEASE) {
        left_mouse_down_ = false;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
      }
    }
    if (action == GLFW_PRESS) {
      keys_[button] = true;
    } else if (action == GLFW_RELEASE) {
      keys_[button] = false;
    }
    for (auto f : user_mouse_button_callbacks_) {
      f(button, action);
    }
  }

  bool is_pressed(int key) {
    return keys_[key];
  }

  float last_x() {
    return last_x_;
  }

  float last_y() {
    return last_y_;
  }

  void add_key_callback(std::function<void(int, int)> f) {
    user_key_callbacks_.push_back(f);
  }
  void add_mouse_pos_callback(std::function<void(double, double)> f) {
    user_mouse_pos_callbacks_.push_back(f);
  }
  void add_mouse_button_callback(std::function<void(int, int)> f) {
    user_mouse_button_callbacks_.push_back(f);
  }

  InputHandler() : keys_(1024, false) {
  }

 private:
  bool first_mouse_ = true;

  bool left_mouse_down_ = false;

  std::vector<bool> keys_;
  float last_x_ = 0;
  float last_y_ = 0;

  std::vector<std::function<void(int, int)>> user_key_callbacks_;
  std::vector<std::function<void(double, double)>> user_mouse_pos_callbacks_;
  std::vector<std::function<void(int, int)>> user_mouse_button_callbacks_;
};

TI_UI_NAMESPACE_END
