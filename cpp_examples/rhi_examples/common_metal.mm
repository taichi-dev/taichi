#include "common_metal.h"

#include <assert.h>
#import <simd/simd.h>
#include <stdio.h>
#include <stdlib.h>

using namespace taichi::lang;

static void glfw_error_callback(int error, const char *description) {
  TI_WARN("GLFW Error {}: {}", error, description);
}

App::App(int width, int height, const std::string &title) {
  TI_INFO("Creating App '{}' of {}x{}", title, width, height);

  TI_ASSERT(metal::is_metal_api_available());

  rhi_metal_device = metal::MetalDevice::create();
  MTLDevice_id mtl_device = rhi_metal_device->mtl_device();

  if (!mtl_device)
    TI_ERROR("failed to init Metal Device");

  if (glfwInit()) {
    TI_INFO("Initialized GLFW");

    glfwSetErrorCallback(glfw_error_callback);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfw_window =
        glfwCreateWindow(width, height, "Sample Window", nullptr, nullptr);
    if (!glfw_window) {
      glfwTerminate();
      TI_ERROR("failed to init GLFW Window");
    }
    TI_INFO("Initialized GLFW Window");
  } else {
    TI_ERROR("failed to init GLFW");
  }

  SurfaceConfig config;
  config.width = width;
  config.height = height;

  surface = rhi_metal_device->create_surface(config);

  metal::MetalSurface* mtl_surf = dynamic_cast<metal::MetalSurface*> (surface.get());

  NSWindow *nswin = glfwGetCocoaWindow(glfw_window);
  nswin.contentView.layer = mtl_surf->mtl_layer();
  nswin.contentView.wantsLayer = YES;

  
}

App::~App() {
  surface.reset();
  glfwDestroyWindow(glfw_window);
  glfwTerminate();
}

void App::run() {
  while (!glfwWindowShouldClose(glfw_window)) {
    auto image_available_semaphore = surface->acquire_next_image();

    glfwPollEvents();

    surface->present_image(render_loop(image_available_semaphore));
  }
}
