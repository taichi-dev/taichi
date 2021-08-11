#include "taichi/ui/backends/vulkan/window.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

Window::Window(AppConfig config) : WindowBase(config) {
  app_context_.config = config;
  init();
}

void Window::init() {
  init_window();
  init_vulkan();
  gui_.init(&app_context_, glfw_window_);
  prepare_for_next_frame();
}

void Window::show() {
  draw_frame();
  present_frame();
  WindowBase::show();
  prepare_for_next_frame();
}

void Window::prepare_for_next_frame() {
  app_context_.swap_chain.update_image_index();
  canvas_->prepare_for_next_frame();
  gui_.prepare_for_next_frame();
}

CanvasBase *Window::get_canvas() {
  return canvas_.get();
}

GuiBase *Window::GUI() {
  return &gui_;
}

void Window::init_window() {
  glfwSetFramebufferSizeCallback(glfw_window_, framebuffer_resize_callback);
}

void Window::framebuffer_resize_callback(GLFWwindow *glfw_window_,
                                         int width,
                                         int height) {
  auto window =
      reinterpret_cast<Window *>(glfwGetWindowUserPointer(glfw_window_));
  window->recreate_swap_chain();
}

void Window::init_vulkan() {
  app_context_.glfw_window = glfw_window_;
  app_context_.init();
  canvas_ = std::make_unique<Canvas>(&app_context_);
}

void Window::cleanup_swap_chain() {
  canvas_->cleanup_swap_chain();
  app_context_.cleanup_swap_chain();
}

void Window::cleanup() {
  gui_.cleanup();
  cleanup_swap_chain();

  canvas_->cleanup();

  app_context_.cleanup();

  glfwTerminate();
}

void Window::recreate_swap_chain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(glfw_window_, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(glfw_window_, &width, &height);
    glfwWaitEvents();
  }
  app_context_.config.width = width;
  app_context_.config.height = height;

  vkDeviceWaitIdle(app_context_.device());

  cleanup_swap_chain();

  app_context_.recreate_swap_chain();

  canvas_->recreate_swap_chain();
}

void Window::draw_frame() {
  canvas_->draw_frame(&gui_);
}

void Window::present_frame() {
  app_context_.swap_chain.present_frame();
  if (app_context_.swap_chain.requires_recreate) {
    recreate_swap_chain();
  }
}

Window::~Window() {
  cleanup();
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
