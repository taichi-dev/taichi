#include "gui.h"
#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/ui/ggui/app_context.h"

// ImGui backends
#include "imgui.h"
#include "imgui_impl_vulkan.h"

// For VulkanLoader::instance() usage
using namespace taichi::lang::vulkan;
using namespace taichi::lang;

namespace taichi::ui {
namespace vulkan {

/**
 * The newer ImGui_ImplVulkan_LoadFunctions(...) signature requires:
 *    bool ImGui_ImplVulkan_LoadFunctions(
 *       uint32_t                         api_version,
 *       PFN_vkVoidFunction(*loader_func)(const char* function_name, void*
 * user_data), void*                            user_data = nullptr
 *    );
 *
 * So we provide a loader that matches the PFN signature.
 */
PFN_vkVoidFunction load_vk_function_for_gui(const char *name, void *user_data) {
  // If you need 'user_data', you can cast it to your VkInstance or device.
  // For now, we just ignore it here and load via Volk.
  return VulkanLoader::instance().load_function(name);
}

Gui::Gui(AppContext *app_context, SwapChain *swap_chain, TaichiWindow *window) {
  app_context_ = app_context;
  swap_chain_ = swap_chain;

  create_descriptor_pool();

  IMGUI_CHECKVERSION();
  imgui_context_ = ImGui::CreateContext();
  [[maybe_unused]] ImGuiIO &io = ImGui::GetIO();

  ImGui::StyleColorsDark();

  if (app_context->config.show_window) {
#ifdef ANDROID
    ImGui_ImplAndroid_Init(window);
    widthBeforeDPIScale = (int)ANativeWindow_getWidth(window);
    heightBeforeDPIScale = (int)ANativeWindow_getHeight(window);
#else
    ImGui_ImplGlfw_InitForVulkan(window, true);
    glfwGetWindowSize(window, &widthBeforeDPIScale, &heightBeforeDPIScale);
#endif
  } else {
    widthBeforeDPIScale = app_context->config.width;
    heightBeforeDPIScale = app_context->config.height;
  }
}

void Gui::init_render_resources(VkRenderPass render_pass) {
  auto &device =
      static_cast<taichi::lang::vulkan::VulkanDevice &>(app_context_->device());

  // -------------------------------------------------
  // 1) Load Vulkan function pointers using new API
  // -------------------------------------------------
  // The new ImGui_ImplVulkan_LoadFunctions(...) requires
  // at least an API version and a PFN loader. The final "user_data" is
  // optional.
  ImGui_ImplVulkan_LoadFunctions(
      device.vk_caps().vk_api_version,
      load_vk_function_for_gui, /*user_data=*/nullptr
  );

  // -------------------------------------------------
  // 2) Prepare ImGui_ImplVulkan_InitInfo
  // -------------------------------------------------
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = device.vk_instance();
  init_info.PhysicalDevice = device.vk_physical_device();
  init_info.Device = device.vk_device();
  init_info.QueueFamily = device.graphics_queue_family_index();
  init_info.Queue = device.graphics_queue();
  init_info.PipelineCache = VK_NULL_HANDLE;
  init_info.DescriptorPool = descriptor_pool_;
  // The new ImGui_ImplVulkan_Init(...) function no longer takes 'render_pass'
  // as a separate argument; you must put it in init_info:
  init_info.RenderPass = render_pass;
  init_info.Subpass = 0;  // If you're not using subpasses, leave as 0
  init_info.Allocator = VK_NULL_HANDLE;
  init_info.MinImageCount = swap_chain_->surface().get_image_count();
  init_info.ImageCount = swap_chain_->surface().get_image_count();
  init_info.CheckVkResultFn = nullptr;  // or your own error-check function
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

  // -------------------------------------------------
  // 3) Call ImGui_ImplVulkan_Init with single param
  // -------------------------------------------------
  ImGui_ImplVulkan_Init(&init_info);

  render_pass_ = render_pass;

  // https://github.com/ocornut/imgui/blob/faa03031b4cdf34fe9174c4e73dd769a5b41fda5/backends/imgui_impl_vulkan.cpp#L47
  // We no longer need to explicitly upload fonts

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

  [[maybe_unused]] VkResult err = vkCreateDescriptorPool(
      static_cast<taichi::lang::vulkan::VulkanDevice &>(app_context_->device())
          .vk_device(),
      &pool_info, VK_NULL_HANDLE, &descriptor_pool_);
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
    // For headless mode, we must manually set io.DisplaySize
    auto w = app_context_->config.width;
    auto h = app_context_->config.height;
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)w, (float)h);
  }

  ImGui::NewFrame();
  is_empty_ = true;
}

bool Gui::initialized() {
  return (render_pass_ != VK_NULL_HANDLE);
}

float Gui::abs_x(float x) {
  return x * widthBeforeDPIScale;
}
float Gui::abs_y(float y) {
  return y * heightBeforeDPIScale;
}

void Gui::begin(const std::string &name,
                float x,
                float y,
                float width,
                float height) {
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

void Gui::text(const std::string &text) {
  if (!initialized()) {
    return;
  }
  ImGui::TextUnformatted(text.c_str());
}

void Gui::text(const std::string &text, glm::vec3 color) {
  if (!initialized()) {
    return;
  }
  ImGui::TextColored(ImVec4(color[0], color[1], color[2], 1.0f), "%s",
                     text.c_str());
}

bool Gui::checkbox(const std::string &name, bool old_value) {
  if (!initialized()) {
    return old_value;
  }
  ImGui::Checkbox(name.c_str(), &old_value);
  return old_value;
}

int Gui::slider_int(const std::string &name,
                    int old_value,
                    int minimum,
                    int maximum) {
  if (!initialized()) {
    return old_value;
  }
  ImGui::SliderInt(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}

float Gui::slider_float(const std::string &name,
                        float old_value,
                        float minimum,
                        float maximum) {
  if (!initialized()) {
    return old_value;
  }
  ImGui::SliderFloat(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}

glm::vec3 Gui::color_edit_3(const std::string &name, glm::vec3 old_value) {
  if (!initialized()) {
    return old_value;
  }
  ImGui::ColorEdit3(name.c_str(), (float *)&old_value);
  return old_value;
}

bool Gui::button(const std::string &text) {
  if (!initialized()) {
    return false;
  }
  return ImGui::Button(text.c_str());
}

void Gui::draw(taichi::lang::CommandList *cmd_list) {
  ImGui::Render();
  ImDrawData *draw_data = ImGui::GetDrawData();

  VkCommandBuffer buffer =
      static_cast<VulkanCommandList *>(cmd_list)->vk_command_buffer()->buffer;

  // Note: The renderpass has started & will finish in the caller of this function
  // This call remains the same in the new API
  ImGui_ImplVulkan_RenderDrawData(draw_data, buffer);
}

void Gui::cleanup_render_resources() {
  vkDestroyDescriptorPool(
      static_cast<taichi::lang::vulkan::VulkanDevice &>(app_context_->device())
          .vk_device(),
      descriptor_pool_, nullptr);

  if (initialized()) {
    ImGui_ImplVulkan_Shutdown();
  }
  render_pass_ = VK_NULL_HANDLE;
}

Gui::~Gui() {
  if (app_context_->config.show_window) {
#ifdef ANDROID
    ImGui_ImplAndroid_Shutdown();
#else
    ImGui_ImplGlfw_Shutdown();
#endif
  }
  cleanup_render_resources();
  ImGui::DestroyContext(imgui_context_);
}

bool Gui::is_empty() {
  return is_empty_;
}

}  // namespace vulkan
}  // namespace taichi::ui
