#pragma once

#define GLFW_INCLUDE_NONE
#import <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#import <GLFW/glfw3native.h>
#include "glm/glm.hpp"

#if defined(__APPLE__) && defined(__OBJC__)
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#define DEFINE_METAL_ID_TYPE(x) typedef id<x> x##_id;
#else
#define DEFINE_METAL_ID_TYPE(x) typedef struct x##_t *x##_id;
#endif

DEFINE_METAL_ID_TYPE(MTLDevice);
DEFINE_METAL_ID_TYPE(MTLBuffer);
DEFINE_METAL_ID_TYPE(MTLTexture);
DEFINE_METAL_ID_TYPE(MTLSamplerState);
DEFINE_METAL_ID_TYPE(MTLLibrary);
DEFINE_METAL_ID_TYPE(MTLFunction);
DEFINE_METAL_ID_TYPE(MTLComputePipelineState);
DEFINE_METAL_ID_TYPE(MTLCommandQueue);
DEFINE_METAL_ID_TYPE(MTLCommandBuffer);
DEFINE_METAL_ID_TYPE(MTLBlitCommandEncoder);
DEFINE_METAL_ID_TYPE(MTLComputeCommandEncoder);
DEFINE_METAL_ID_TYPE(CAMetalDrawable);

#undef DEFINE_METAL_ID_TYPE

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