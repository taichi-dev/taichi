#pragma once
#include <memory>
#include <functional>
#include <vector>
#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

class InputHandler {
 public:
  std::vector<bool> keys;
  float last_x = 0;
  float last_y = 0;

  std::vector<std::function<void(int, int)>> user_key_callbacks;
  std::vector<std::function<void(double, double)>> use_mouse_pos_callbacks;
  std::vector<std::function<void(int, int)>> user_mouse_button_callbacks;

  void key_callback(GLFWwindow *window,
                    int key,
                    int scancode,
                    int action,
                    int mode) {
    if (action == GLFW_PRESS) {
      keys[key] = true;
    } else if (action == GLFW_RELEASE) {
      keys[key] = false;
    }
    for (auto f : user_key_callbacks) {
      f(key, action);
    }
  }

  void mouse_pos_callback(GLFWwindow *window, double xpos, double ypos) {
    if (first_mouse_) {
      last_x = xpos;
      last_y = ypos;
      first_mouse_ = false;
    }

    last_x = xpos;
    last_y = ypos;

    for (auto f : use_mouse_pos_callbacks) {
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
      keys[button] = true;
    } else if (action == GLFW_RELEASE) {
      keys[button] = false;
    }
    for (auto f : user_mouse_button_callbacks) {
      f(button, action);
    }
  }

  InputHandler() : keys(1024, false) {
  }

 private:
  bool first_mouse_ = true;

  bool left_mouse_down_ = false;
};

TI_UI_NAMESPACE_END
