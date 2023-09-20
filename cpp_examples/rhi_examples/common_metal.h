#pragma once

#define GLFW_INCLUDE_NONE
#import <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#import <GLFW/glfw3native.h>
#include "glm/glm.hpp"

#include "taichi/rhi/metal/metal_api.h"
#include "taichi/rhi/metal/metal_device.h"

using namespace taichi::lang;

class App {
 public:
  explicit App(int width, int height, const std::string &title);
  virtual ~App();

  virtual std::vector<StreamSemaphore> render_loop(
      StreamSemaphore image_available_semaphore) {
    return {};
  }

  void run();

  GLFWwindow *glfw_window;
  metal::MetalDevice *device;

  std::unique_ptr<Surface> surface;
};
