#include "gui.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"

using namespace taichi::lang::vulkan;

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

PFN_vkVoidFunction load_vk_function_for_gui(const char *name, void *userData) {
  auto result = VulkanLoader::instance().load_function(name);

  return result;
}

void Gui::init(AppContext *app_context, GLFWwindow *window) {
  app_context_ = app_context;

  create_descriptor_pool();

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable
  // Keyboard Controls io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; //
  // Enable Gamepad Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForVulkan(window, true);
  ImGui_ImplVulkan_LoadFunctions(
      load_vk_function_for_gui);  // this is becaus we're using volk.
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = app_context_->instance();
  init_info.PhysicalDevice = app_context_->physical_device();
  init_info.Device = app_context_->device();
  init_info.QueueFamily =
      app_context_->queue_family_indices().graphics_family.value();
  init_info.Queue = app_context_->graphics_queue();
  init_info.PipelineCache = VK_NULL_HANDLE;
  init_info.DescriptorPool = descriptor_pool_;
  init_info.Allocator = VK_NULL_HANDLE;
  ;
  init_info.MinImageCount = app_context_->swap_chain.swap_chain_images.size();
  init_info.ImageCount = app_context_->swap_chain.swap_chain_images.size();
  ImGui_ImplVulkan_Init(&init_info, app_context_->render_pass());

  // Load Fonts
  // - If no fonts are loaded, dear imgui will use the default font. You can
  // also load multiple fonts and use ImGui::PushFont()/PopFont() to select
  // them.
  // - AddFontFromFileTTF() will return the ImFont* so you can store it if you
  // need to select the font among multiple.
  // - If the file cannot be loaded, the function will return NULL. Please
  // handle those errors in your application (e.g. use an assertion, or display
  // an error and quit).
  // - The fonts will be rasterized at a given size (w/ oversampling) and stored
  // into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which
  // ImGui_ImplXXXX_NewFrame below will call.
  // - Read 'docs/FONTS.md' for more instructions and details.
  // - Remember that in C/C++ if you want to include a backslash \ in a string
  // literal you need to write a double backslash \\ !
  // io.Fonts->AddFontDefault();
  // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
  // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
  // io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
  // io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
  // ImFont* font =
  // io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f,
  // NULL, io.Fonts->GetGlyphRangesJapanese()); IM_ASSERT(font != NULL);

  // Upload Fonts
  {
    // Use any command queue
    VkCommandBuffer command_buffer = begin_single_time_commands(
        app_context->command_pool(), app_context->device());

    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

    VkSubmitInfo end_info = {};
    end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    end_info.commandBufferCount = 1;
    end_info.pCommandBuffers = &command_buffer;
    vkEndCommandBuffer(command_buffer);
    vkQueueSubmit(app_context->graphics_queue(), 1, &end_info, VK_NULL_HANDLE);

    vkDeviceWaitIdle(app_context->device());
    vkFreeCommandBuffers(app_context->device(), app_context->command_pool(), 1,
                         &command_buffer);
    ImGui_ImplVulkan_DestroyFontUploadObjects();
  }
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
  VkResult err = vkCreateDescriptorPool(app_context_->device(), &pool_info,
                                        VK_NULL_HANDLE, &descriptor_pool_);
}

void Gui::prepare_for_next_frame() {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  is_empty = true;
}

float Gui::abs_x(float x) {
  return x * app_context_->swap_chain.swap_chain_extent.width;
}
float Gui::abs_y(float y) {
  return y * app_context_->swap_chain.swap_chain_extent.height;
}

void Gui::begin(std::string name, float x, float y, float width, float height) {
  ImGui::SetNextWindowPos(ImVec2(abs_x(x), abs_y(y)), ImGuiCond_Once);
  ImGui::SetNextWindowSize(ImVec2(abs_x(width), abs_y(height)), ImGuiCond_Once);
  ImGui::Begin(name.c_str());
  is_empty = false;
}
void Gui::end() {
  ImGui::End();
}
void Gui::text(std::string text) {
  ImGui::Text(text.c_str());
}
bool Gui::checkbox(std::string name, bool old_value) {
  ImGui::Checkbox(name.c_str(), &old_value);
  return old_value;
}
float Gui::slider_float(std::string name,
                        float old_value,
                        float minimum,
                        float maximum) {
  ImGui::SliderFloat(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}
glm::vec3 Gui::color_edit_3(std::string name, glm::vec3 old_value) {
  ImGui::ColorEdit3(name.c_str(), (float *)&old_value);
  return old_value;
}
bool Gui::button(std::string text) {
  return ImGui::Button(text.c_str());
}

void Gui::draw(VkCommandBuffer &command_buffer) {
  // Rendering
  ImGui::Render();
  ImDrawData *draw_data = ImGui::GetDrawData();

  ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);
}

void Gui::cleanup() {
  vkDestroyDescriptorPool(app_context_->device(), descriptor_pool_, nullptr);

  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
