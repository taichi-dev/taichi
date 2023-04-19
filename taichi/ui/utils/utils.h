

#pragma once

#include <string>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
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
#include <unordered_map>

#ifdef _WIN64
#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>
#include <VersionHelpers.h>
#define _USE_MATH_DEFINES
#endif

#ifdef _WIN64
#define VK_USE_PLATFORM_WIN32_KHR 1
#endif

#include "taichi/rhi/vulkan/vulkan_common.h"
#include "taichi/rhi/common/window_system.h"
#include "taichi/common/filesystem.hpp"

#include <stdarg.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace taichi::ui {

#define RHI_VERIFY(rhi_call)                                    \
  {                                                             \
    taichi::lang::RhiResult r = rhi_call;                       \
    TI_ASSERT_INFO(r == taichi::lang::RhiResult::success,       \
                   "`{}` failed, error {}", #rhi_call, int(r)); \
  }

#ifdef TI_WITH_GLFW
inline GLFWwindow *create_glfw_window_(const std::string &name,
                                       int screenWidth,
                                       int screenHeight,
                                       int window_pos_x,
                                       int window_pos_y,
                                       bool vsync) {
  if (!taichi::lang::window_system::glfw_context_acquire()) {
    printf("cannot initialize GLFW\n");
    exit(EXIT_FAILURE);
  }
  GLFWwindow *window;

  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window = glfwCreateWindow(screenWidth, screenHeight, name.c_str(), nullptr,
                            nullptr);

  if (!window) {
    taichi::lang::window_system::glfw_context_release();
    exit(EXIT_FAILURE);
  }

  // Reset the window hints to default
  glfwDefaultWindowHints();

  glfwSetWindowPos(window, window_pos_x, window_pos_y);

  glfwShowWindow(window);
  // Invalid for Vulkan
  /*
  if (vsync) {
    glfwSwapInterval(1);
  } else {
    glfwSwapInterval(0);
  }
  */
  return window;
}

struct Keys {
#define DEFINE_KEY(name) static inline const std::string name = #name

  DEFINE_KEY(Shift);
  DEFINE_KEY(Alt);
  DEFINE_KEY(Control);
  DEFINE_KEY(Escape);
  DEFINE_KEY(Return);
  DEFINE_KEY(Tab);
  DEFINE_KEY(BackSpace);
  static inline const std::string Space = " ";
  DEFINE_KEY(Up);
  DEFINE_KEY(Down);
  DEFINE_KEY(Left);
  DEFINE_KEY(Right);
  DEFINE_KEY(CapsLock);
  DEFINE_KEY(LMB);
  DEFINE_KEY(MMB);
  DEFINE_KEY(RMB);
#undef DEFINE_KEY
};

inline std::unordered_map<std::string, int> get_keys_map() {
  static std::unordered_map<std::string, int> keys = {
      {Keys::Shift, GLFW_KEY_LEFT_SHIFT},
      {Keys::Alt, GLFW_KEY_LEFT_ALT},
      {Keys::Control, GLFW_KEY_LEFT_CONTROL},
      {Keys::Escape, GLFW_KEY_ESCAPE},
      {Keys::Return, GLFW_KEY_ENTER},
      {Keys::Tab, GLFW_KEY_TAB},
      {Keys::BackSpace, GLFW_KEY_BACKSPACE},
      {Keys::Space, GLFW_KEY_SPACE},
      {Keys::Up, GLFW_KEY_UP},
      {Keys::Down, GLFW_KEY_DOWN},
      {Keys::Left, GLFW_KEY_LEFT},
      {Keys::Right, GLFW_KEY_RIGHT},
      {Keys::CapsLock, GLFW_KEY_CAPS_LOCK},
      {Keys::LMB, GLFW_MOUSE_BUTTON_LEFT},
      {Keys::MMB, GLFW_MOUSE_BUTTON_MIDDLE},
      {Keys::RMB, GLFW_MOUSE_BUTTON_RIGHT}};
  return keys;
}

inline std::unordered_map<int, std::string> get_inv_keys_map() {
  auto keys = get_keys_map();
  std::unordered_map<int, std::string> keys_inv;
  for (auto kv : keys) {
    keys_inv[kv.second] = kv.first;
  }
  keys_inv[GLFW_KEY_RIGHT_SHIFT] = Keys::Shift;
  keys_inv[GLFW_KEY_RIGHT_CONTROL] = Keys::Control;
  keys_inv[GLFW_KEY_RIGHT_ALT] = Keys::Alt;
  return keys_inv;
}

inline int buttom_name_to_id(const std::string &name) {
  if (name.size() == 1) {
    char c = name[0];
    if (c >= 'a' && c <= 'z') {
      c = c - ('a' - 'A');
      return (int)c;
    }
  }

  auto keys = get_keys_map();

  if (keys.find(name) != keys.end()) {
    return keys.at(name);
  } else {
    throw std::runtime_error(std::string("unrecognized name: ") + name);
  }
}

inline std::string button_id_to_name(int id) {
  if (id >= 'A' && id <= 'Z') {
    char c = id + ('a' - 'A');
    std::string name;
    name += c;
    return name;
  }
  auto keys = get_inv_keys_map();

  if (keys.find(id) != keys.end()) {
    return keys.at(id);
  } else {
    throw std::runtime_error(std::string("unrecognized id: ") +
                             std::to_string(id));
  }
}
#endif

inline std::vector<char> read_file(const std::string &filename) {
  std::ifstream file(std::filesystem::path{filename},
                     std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error(filename + " failed to open file!");
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}

}  // namespace taichi::ui
