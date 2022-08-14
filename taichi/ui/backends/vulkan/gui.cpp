#include "gui.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
#include "taichi/ui/backends/vulkan/app_context.h"

using namespace taichi::lang::vulkan;
using namespace taichi::lang;

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

PFN_vkVoidFunction load_vk_function_for_gui(const char *name, void *userData) {
  auto result = VulkanLoader::instance().load_function(name);

  return result;
}

Gui::Gui(AppContext *app_context, SwapChain *swap_chain, TaichiWindow *window) {
  app_context_ = app_context;
  swap_chain_ = swap_chain;

  create_descriptor_pool();

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  [[maybe_unused]] ImGuiIO &io = ImGui::GetIO();

  ImGui::StyleColorsDark();

  if (app_context->config.show_window) {
#ifdef ANDROID
    ImGui_ImplAndroid_Init(window);
#else
    ImGui_ImplGlfw_InitForVulkan(window, true);
#endif
  }
}

void Gui::init_render_resources(VkRenderPass render_pass) {
  ImGui_ImplVulkan_LoadFunctions(
      load_vk_function_for_gui);  // this is because we're using volk.

  auto &device = app_context_->device();

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = device.vk_instance();
  init_info.PhysicalDevice = device.vk_physical_device();
  init_info.Device = device.vk_device();
  init_info.QueueFamily = device.graphics_queue_family_index();
  init_info.Queue = device.graphics_queue();
  init_info.PipelineCache = VK_NULL_HANDLE;
  init_info.DescriptorPool = descriptor_pool_;
  init_info.Allocator = VK_NULL_HANDLE;
  init_info.MinImageCount = swap_chain_->surface().get_image_count();
  init_info.ImageCount = swap_chain_->surface().get_image_count();
  ImGui_ImplVulkan_Init(&init_info, render_pass);
  render_pass_ = render_pass;

  // Upload Fonts
  {
    auto stream = device.get_graphics_stream();
    std::unique_ptr<CommandList> cmd_list = stream->new_command_list();
    VkCommandBuffer command_buffer =
        static_cast<VulkanCommandList *>(cmd_list.get())
            ->vk_command_buffer()
            ->buffer;

    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

    stream->submit_synced(cmd_list.get());
    ImGui_ImplVulkan_DestroyFontUploadObjects();
  }
  prepare_for_next_frame();
}

void Gui::create_descriptor_pool() {
  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
  pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;
  [[maybe_unused]] VkResult err =
      vkCreateDescriptorPool(app_context_->device().vk_device(), &pool_info,
                             VK_NULL_HANDLE, &descriptor_pool_);
}

void Gui::prepare_for_next_frame() {
  if (render_pass_ == VK_NULL_HANDLE) {
    return;
  }
  ImGui_ImplVulkan_NewFrame();
  if (app_context_->config.show_window) {
#ifdef ANDROID
    ImGui_ImplAndroid_NewFrame();
#else
    ImGui_ImplGlfw_NewFrame();
#endif
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

bool Gui::initialized() {
  return render_pass_ != VK_NULL_HANDLE;
}

float Gui::abs_x(float x) {
  return x * app_context_->config.width;
}
float Gui::abs_y(float y) {
  return y * app_context_->config.height;
}

void Gui::begin(std::string name, float x, float y, float width, float height) {
  if (!initialized()) {
    return;
  }
  ImGui::SetNextWindowPos(ImVec2(abs_x(x), abs_y(y)), ImGuiCond_Once);
  ImGui::SetNextWindowSize(ImVec2(abs_x(width), abs_y(height)), ImGuiCond_Once);
  ImGui::Begin(name.c_str());
  is_empty_ = false;
}
void Gui::end() {
  if (!initialized()) {
    return;
  }
  ImGui::End();
}
void Gui::text(std::string text) {
  if (!initialized()) {
    return;
  }
  ImGui::Text("%s", text.c_str());
}
bool Gui::checkbox(std::string name, bool old_value) {
  if (!initialized()) {
    return old_value;
  }
  ImGui::Checkbox(name.c_str(), &old_value);
  return old_value;
}
int Gui::slider_int(std::string name, int old_value, int minimum, int maximum) {
  if (!initialized()) {
    return old_value;
  }
  ImGui::SliderInt(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}
float Gui::slider_float(std::string name,
                        float old_value,
                        float minimum,
                        float maximum) {
  if (!initialized()) {
    return old_value;
  }
  ImGui::SliderFloat(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}
glm::vec3 Gui::color_edit_3(std::string name, glm::vec3 old_value) {
  if (!initialized()) {
    return old_value;
  }
  ImGui::ColorEdit3(name.c_str(), (float *)&old_value);
  return old_value;
}
bool Gui::button(std::string text) {
  if (!initialized()) {
    return false;
  }
  return ImGui::Button(text.c_str());
}

void Gui::draw(taichi::lang::CommandList *cmd_list) {
  // Rendering
  ImGui::Render();
  ImDrawData *draw_data = ImGui::GetDrawData();

  VkCommandBuffer buffer =
      static_cast<VulkanCommandList *>(cmd_list)->vk_command_buffer()->buffer;

  ImGui_ImplVulkan_RenderDrawData(draw_data, buffer);
}

void Gui::cleanup_render_resources() {
  vkDestroyDescriptorPool(app_context_->device().vk_device(), descriptor_pool_,
                          nullptr);

  if (initialized()) {
    ImGui_ImplVulkan_Shutdown();
  }
  render_pass_ = VK_NULL_HANDLE;
}

void Gui::cleanup() {
  if (app_context_->config.show_window) {
#ifdef ANDROID
    ImGui_ImplAndroid_Shutdown();
#else
    ImGui_ImplGlfw_Shutdown();
#endif
  }
  cleanup_render_resources();
  ImGui::DestroyContext();
}

Gui::~Gui() {
  cleanup();
}
bool Gui::is_empty() {
  return is_empty_;
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
