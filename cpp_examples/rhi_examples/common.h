#pragma once

#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_common.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"

#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"

using namespace taichi::lang;

static void glfw_error_callback(int code, const char *description) {
  TI_WARN("GLFW Error {}: {}", code, description);
}

std::vector<std::string> get_required_instance_extensions() {
  std::vector<std::string> extensions;

  uint32_t glfw_ext_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  for (int i = 0; i < glfw_ext_count; ++i) {
    extensions.push_back(glfw_extensions[i]);
  }
  // VulkanDeviceCreator will check that these are supported
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  return extensions;
}

std::vector<std::string> get_required_device_extensions() {
  static std::vector<std::string> extensions{
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  };

  return extensions;
}

class App {
 public:
  App(int width, int height, const std::string &title) {
    if (glfwInit()) {
      glfwSetErrorCallback(glfw_error_callback);

      glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      glfw_window =
          glfwCreateWindow(width, height, "Sample Window", nullptr, nullptr);

      if (glfwVulkanSupported() != GLFW_TRUE) {
        TI_WARN("GLFW reports no Vulkan support");
      }
    } else {
      throw std::runtime_error("failed to init GLFW");
    }

    {
      vulkan::VulkanDeviceCreator::Params evd_params;
      evd_params.api_version = std::nullopt;
      evd_params.additional_instance_extensions =
          get_required_instance_extensions();
      evd_params.additional_device_extensions =
          get_required_device_extensions();
      evd_params.is_for_ui = true;
      evd_params.surface_creator = [&](VkInstance instance) -> VkSurfaceKHR {
        VkSurfaceKHR surface = VK_NULL_HANDLE;

        if (glfwCreateWindowSurface(instance, glfw_window, nullptr, &surface) !=
            VK_SUCCESS) {
          throw std::runtime_error("failed to create window surface!");
        }
        return surface;
      };

      device_creator =
          std::make_unique<vulkan::VulkanDeviceCreator>(evd_params);
      device = device_creator->device();
    }

    {
      SurfaceConfig config;
      config.window_handle = glfw_window;
      config.native_surface_handle = device_creator->get_surface();

      surface = device->create_surface(config);
    }
  }

  virtual ~App() {
    surface.reset();
    device_creator.reset();
    glfwDestroyWindow(glfw_window);
    glfwTerminate();
  }

  virtual std::vector<StreamSemaphore> render_loop(
      StreamSemaphore image_available_semaphore) {
    return {};
  }

  void run() {
    while (!glfwWindowShouldClose(glfw_window)) {
      auto image_available_semaphore = surface->acquire_next_image();

      glfwPollEvents();

      surface->present_image(render_loop(image_available_semaphore));
    }
  }

 public:
  // Owned
  GLFWwindow *glfw_window;
  std::unique_ptr<vulkan::VulkanDeviceCreator> device_creator;
  std::unique_ptr<Surface> surface;

  // Weak references
  vulkan::VulkanDevice *device;
};
