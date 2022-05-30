#include "taichi/ui/backends/vulkan/window.h"
#include "taichi/program/callable.h"

using taichi::lang::Program;

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

Window::Window(Program *prog, const AppConfig &config) : WindowBase(config) {
  init(prog, config);
}

void Window::init(Program *prog, const AppConfig &config) {
  if (config_.show_window) {
    glfwSetFramebufferSizeCallback(glfw_window_, framebuffer_resize_callback);
  }

  renderer_ = std::make_unique<Renderer>();
  renderer_->init(prog, glfw_window_, config);
  canvas_ = std::make_unique<Canvas>(renderer_.get());
  gui_ = std::make_unique<Gui>(&renderer_->app_context(),
                               &renderer_->swap_chain(), glfw_window_);

  prepare_for_next_frame();
}

void Window::show() {
  if (!drawn_frame_) {
    draw_frame();
  }
  present_frame();
  WindowBase::show();
  prepare_for_next_frame();
}

void Window::prepare_for_next_frame() {
  renderer_->prepare_for_next_frame();
  gui_->prepare_for_next_frame();
  drawn_frame_ = false;
}

CanvasBase *Window::get_canvas() {
  return canvas_.get();
}

GuiBase *Window::GUI() {
  return gui_.get();
}

void Window::framebuffer_resize_callback(GLFWwindow *glfw_window_,
                                         int width,
                                         int height) {
  auto window =
      reinterpret_cast<Window *>(glfwGetWindowUserPointer(glfw_window_));
  window->resize();
}

void Window::resize() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(glfw_window_, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(glfw_window_, &width, &height);
    glfwWaitEvents();
  }
  renderer_->app_context().config.width = width;
  renderer_->app_context().config.height = height;

  renderer_->swap_chain().resize(width, height);

  // config_.width and config_.height are used for computing relative mouse
  // positions, so they need to be updated once the window is resized.
  config_.width = width;
  config_.height = height;
}

void Window::draw_frame() {
  renderer_->draw_frame(gui_.get());
  drawn_frame_ = true;
}

void Window::present_frame() {
  renderer_->swap_chain().surface().present_image(
      {renderer_->get_render_complete_semaphore()});
}

Window::~Window() {
  gui_->cleanup();
  renderer_->cleanup();
  if (config_.show_window) {
    glfwTerminate();
  }
}

void Window::write_image(const std::string &filename) {
  if (!drawn_frame_) {
    draw_frame();
  }
  renderer_->swap_chain().write_image(filename);
  if (!config_.show_window) {
    prepare_for_next_frame();
  }
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
