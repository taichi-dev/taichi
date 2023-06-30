#include "common_metal.h"

#include <assert.h>
#import <simd/simd.h>
#include <stdio.h>
#include <stdlib.h>

static void glfw_error_callback(int error, const char *description) {
  TI_WARN("GLFW Error {}: {}", error, description);
}

CAMetalLayer *layer;

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

  @autoreleasepool {
    NSWindow *nswin = glfwGetCocoaWindow(glfw_window);
    layer = [CAMetalLayer layer];
    layer.device = mtl_device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.drawableSize = CGSizeMake(width, height);
    nswin.contentView.layer = layer;
    nswin.contentView.wantsLayer = YES;
  }

  metal::MetalStream *stream =
      static_cast<metal::MetalStream *>(rhi_metal_device->get_compute_stream());
  command_queue = stream->mtl_command_queue();
}

App::~App() {
  glfwDestroyWindow(glfw_window);
  glfwTerminate();
}

void App::run() {
  while (!glfwWindowShouldClose(glfw_window)) {
    // TODO: Make this use the RHI via a MetalSurface rather than directly
    // sending messages to the Metal Layer
    CAMetalDrawable_id drawable = [layer nextDrawable];
    if (!drawable) continue;
    
    glfwPollEvents();

    MTLCommandBuffer_id cb = [command_queue commandBuffer];
    [cb presentDrawable:drawable];
    [cb commit];
  }
}
