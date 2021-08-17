#include "taichi/ui/backends/vulkan/window.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

Window::Window(const AppConfig &config) : WindowBase(config) {
  init(config);
}

void Window::init(const AppConfig &config) {
  init_window();
  init_vulkan(config);
  gui_.init(renderer_.get(), glfw_window_);
  prepare_for_next_frame();
}

void Window::show() {
  draw_frame();
  present_frame();
  WindowBase::show();
  prepare_for_next_frame();
}

void Window::prepare_for_next_frame() {
  renderer_->prepare_for_next_frame();
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

void Window::init_vulkan(const AppConfig &config) {
  renderer_ = std::make_unique<Renderer>();
  renderer_->init(glfw_window_, config);
  canvas_ = std::make_unique<Canvas>(renderer_.get());
}

void Window::cleanup_swap_chain() {
  renderer_->cleanup_swap_chain();
}

void Window::cleanup() {
  gui_.cleanup();
  cleanup_swap_chain();

  renderer_->cleanup();

  glfwTerminate();
}

void Window::recreate_swap_chain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(glfw_window_, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(glfw_window_, &width, &height);
    glfwWaitEvents();
  }
  renderer_->app_context().config.width = width;
  renderer_->app_context().config.height = height;

  vkDeviceWaitIdle(renderer_->app_context().device());

  cleanup_swap_chain();

  renderer_->recreate_swap_chain();
}

void Window::draw_frame() {
  renderer_->draw_frame(&gui_);
}

void Window::present_frame() {
  renderer_->present_frame();
}

Window::~Window() {
  cleanup();
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
