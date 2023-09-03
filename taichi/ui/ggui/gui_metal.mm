#include "gui_metal.h"
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/swap_chain.h"
#include <imgui_impl_metal.h>

using namespace taichi::lang::metal;
using namespace taichi::lang;

namespace taichi::ui {

namespace vulkan {

GuiMetal::GuiMetal(AppContext *app_context, TaichiWindow *window) {
  app_context_ = app_context;

  IMGUI_CHECKVERSION();
  imgui_context_ = ImGui::CreateContext();
  [[maybe_unused]] ImGuiIO &io = ImGui::GetIO();

  ImGui::StyleColorsDark();

  if (app_context->config.show_window) {
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    glfwGetWindowSize(window, &widthBeforeDPIScale, &heightBeforeDPIScale);
  } else {
    widthBeforeDPIScale = app_context->config.width;
    heightBeforeDPIScale = app_context->config.height;
  }
  auto &device =
      static_cast<taichi::lang::metal::MetalDevice &>(app_context_->device());

  ImGui_ImplMetal_Init(device.mtl_device());
}

void GuiMetal::init_render_resources(void *rpd) {
  current_rpd_ = (__bridge MTLRenderPassDescriptor *)rpd;
}

void GuiMetal::prepare_for_next_frame() {
  if (app_context_->config.show_window) {
    ImGui_ImplGlfw_NewFrame();
  } else {
    // io.DisplaySize is set during ImGui_ImplGlfw_NewFrame()
    // but since we're headless, we do it explicitly here
    auto w = app_context_->config.width;
    auto h = app_context_->config.height;
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)w, (float)h);
  }
  ImGui::NewFrame();
  is_empty_ = true;
}

float GuiMetal::abs_x(float x) { return x * widthBeforeDPIScale; }
float GuiMetal::abs_y(float y) { return y * heightBeforeDPIScale; }

void GuiMetal::begin(const std::string &name, float x, float y, float width,
                     float height) {
  ImGui::SetNextWindowPos(ImVec2(abs_x(x), abs_y(y)), ImGuiCond_Once);
  ImGui::SetNextWindowSize(ImVec2(abs_x(width), abs_y(height)), ImGuiCond_Once);
  ImGui::Begin(name.c_str());
  is_empty_ = false;
}
void GuiMetal::end() { ImGui::End(); }
void GuiMetal::text(const std::string &text) {
  ImGui::Text("%s", text.c_str());
}
void GuiMetal::text(const std::string &text, glm::vec3 color) {
  ImGui::TextColored(ImVec4(color[0], color[1], color[2], 1.0f), "%s",
                     text.c_str());
}
bool GuiMetal::checkbox(const std::string &name, bool old_value) {
  ImGui::Checkbox(name.c_str(), &old_value);
  return old_value;
}
int GuiMetal::slider_int(const std::string &name, int old_value, int minimum,
                         int maximum) {
  ImGui::SliderInt(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}
float GuiMetal::slider_float(const std::string &name, float old_value,
                             float minimum, float maximum) {
  ImGui::SliderFloat(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}
glm::vec3 GuiMetal::color_edit_3(const std::string &name, glm::vec3 old_value) {
  ImGui::ColorEdit3(name.c_str(), (float *)&old_value);
  return old_value;
}
bool GuiMetal::button(const std::string &text) {
  return ImGui::Button(text.c_str());
}

void GuiMetal::draw(taichi::lang::CommandList *cmd_list) {
  ImGui_ImplMetal_NewFrame(current_rpd_);

  // Rendering
  ImGui::Render();

  @autoreleasepool {
    MTLCommandBuffer_id buffer =
        static_cast<MetalCommandList *>(cmd_list)->finalize();

    MTLRenderCommandEncoder_id rce =
        [buffer renderCommandEncoderWithDescriptor:current_rpd_];
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), buffer, rce);
    [rce endEncoding];
  }
}
void GuiMetal::cleanup_render_resources() {
  ImGui_ImplMetal_Shutdown();
  current_rpd_ = nullptr;
}

GuiMetal::~GuiMetal() {
  if (app_context_->config.show_window) {
    ImGui_ImplGlfw_Shutdown();
  }
  cleanup_render_resources();
  ImGui::DestroyContext(imgui_context_);
}

bool GuiMetal::is_empty() { return is_empty_; }

} // namespace vulkan

} // namespace taichi::ui
