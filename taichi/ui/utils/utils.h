

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

  // Function keys
  DEFINE_KEY(F1);
  DEFINE_KEY(F2);
  DEFINE_KEY(F3);
  DEFINE_KEY(F4);
  DEFINE_KEY(F5);
  DEFINE_KEY(F6);
  DEFINE_KEY(F7);
  DEFINE_KEY(F8);
  DEFINE_KEY(F9);
  DEFINE_KEY(F10);
  DEFINE_KEY(F11);
  DEFINE_KEY(F12);

  // Navigation
  DEFINE_KEY(Insert);
  DEFINE_KEY(Delete);
  DEFINE_KEY(Home);
  DEFINE_KEY(End);
  DEFINE_KEY(PageUp);
  DEFINE_KEY(PageDown);

  // Numpad
  DEFINE_KEY(Numpad0);
  DEFINE_KEY(Numpad1);
  DEFINE_KEY(Numpad2);
  DEFINE_KEY(Numpad3);
  DEFINE_KEY(Numpad4);
  DEFINE_KEY(Numpad5);
  DEFINE_KEY(Numpad6);
  DEFINE_KEY(Numpad7);
  DEFINE_KEY(Numpad8);
  DEFINE_KEY(Numpad9);
  DEFINE_KEY(NumpadDecimal);
  DEFINE_KEY(NumpadDivide);
  DEFINE_KEY(NumpadMultiply);
  DEFINE_KEY(NumpadSubtract);
  DEFINE_KEY(NumpadAdd);
  DEFINE_KEY(NumpadEnter);
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
      {Keys::RMB, GLFW_MOUSE_BUTTON_RIGHT},
      // Function keys
      {Keys::F1, GLFW_KEY_F1},
      {Keys::F2, GLFW_KEY_F2},
      {Keys::F3, GLFW_KEY_F3},
      {Keys::F4, GLFW_KEY_F4},
      {Keys::F5, GLFW_KEY_F5},
      {Keys::F6, GLFW_KEY_F6},
      {Keys::F7, GLFW_KEY_F7},
      {Keys::F8, GLFW_KEY_F8},
      {Keys::F9, GLFW_KEY_F9},
      {Keys::F10, GLFW_KEY_F10},
      {Keys::F11, GLFW_KEY_F11},
      {Keys::F12, GLFW_KEY_F12},
      // Navigation
      {Keys::Insert, GLFW_KEY_INSERT},
      {Keys::Delete, GLFW_KEY_DELETE},
      {Keys::Home, GLFW_KEY_HOME},
      {Keys::End, GLFW_KEY_END},
      {Keys::PageUp, GLFW_KEY_PAGE_UP},
      {Keys::PageDown, GLFW_KEY_PAGE_DOWN},
      // Numpad
      {Keys::Numpad0, GLFW_KEY_KP_0},
      {Keys::Numpad1, GLFW_KEY_KP_1},
      {Keys::Numpad2, GLFW_KEY_KP_2},
      {Keys::Numpad3, GLFW_KEY_KP_3},
      {Keys::Numpad4, GLFW_KEY_KP_4},
      {Keys::Numpad5, GLFW_KEY_KP_5},
      {Keys::Numpad6, GLFW_KEY_KP_6},
      {Keys::Numpad7, GLFW_KEY_KP_7},
      {Keys::Numpad8, GLFW_KEY_KP_8},
      {Keys::Numpad9, GLFW_KEY_KP_9},
      {Keys::NumpadDecimal, GLFW_KEY_KP_DECIMAL},
      {Keys::NumpadDivide, GLFW_KEY_KP_DIVIDE},
      {Keys::NumpadMultiply, GLFW_KEY_KP_MULTIPLY},
      {Keys::NumpadSubtract, GLFW_KEY_KP_SUBTRACT},
      {Keys::NumpadAdd, GLFW_KEY_KP_ADD},
      {Keys::NumpadEnter, GLFW_KEY_KP_ENTER},
  };
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
    if (c >= '0' && c <= '9') {
      return (int)c;  // GLFW_KEY_0-9 are ASCII codes 48-57
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
  if (id >= '0' && id <= '9') {
    std::string name;
    name += (char)id;
    return name;
  }
  auto keys = get_inv_keys_map();

  if (keys.find(id) != keys.end()) {
    return keys.at(id);
  } else {
    // Fallback: return "Key_<id>" instead of throwing for unknown keys.
    // This ensures unmapped keys still generate events that users can match on.
    return std::string("Key_") + std::to_string(id);
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
