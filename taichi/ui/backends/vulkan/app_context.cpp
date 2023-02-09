#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
#include "taichi/program/program.h"
#include "taichi/ui/backends/vulkan/vertex.h"

#include <string_view>

namespace taichi::ui {

namespace vulkan {

using namespace vulkan;
using namespace taichi::lang;

namespace {
std::vector<std::string> get_required_instance_extensions() {
#ifdef ANDROID
  std::vector<std::string> extensions;

  extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  return extensions;
#else
  uint32_t glfw_ext_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  std::vector<std::string> extensions;

  for (int i = 0; i < glfw_ext_count; ++i) {
    extensions.push_back(glfw_extensions[i]);
  }

  // VulkanDeviceCreator will check that these are supported
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  return extensions;
#endif
}

std::vector<std::string> get_required_device_extensions() {
  static std::vector<std::string> extensions {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#if !defined(ANDROID)
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
#endif
  };

  return extensions;
}
}  // namespace

void AppContext::init(Program *prog,
                      TaichiWindow *window,
                      const AppConfig &config) {
  taichi_window_ = window;
  prog_ = prog;
  this->config = config;

  auto make_vk_surface = [&](VkInstance instance) -> VkSurfaceKHR {
      VkSurfaceKHR surface = VK_NULL_HANDLE;
#ifdef ANDROID
      VkAndroidSurfaceCreateInfoKHR createInfo{
          .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
          .pNext = nullptr,
          .flags = 0,
          .window = window};

      vkCreateAndroidSurfaceKHR(instance, &createInfo, nullptr, &surface);
#else
      VkResult result = VK_SUCCESS;
      if ((result = glfwCreateWindowSurface(instance, window, nullptr,
                                            &surface)) != VK_SUCCESS) {
        TI_WARN("Failed to create window: error {}", result);
        return nullptr;
      }
#endif
      return surface;
    };

  // Create a Vulkan device if the original configuration is not for Vulkan or
  // there is no active current program (usage from external library for AOT
  // modules for example).
  if (config.ti_arch != Arch::vulkan || prog == nullptr) {
    taichi::lang::vulkan::VulkanDeviceCreator::Params evd_params{};
    evd_params.additional_instance_extensions =
        get_required_instance_extensions();
    evd_params.additional_device_extensions = get_required_device_extensions();
    evd_params.is_for_ui = config.show_window;
    evd_params.surface_creator = make_vk_surface;
    embedded_vulkan_device_ =
        std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
    native_surface_ = embedded_vulkan_device_->get_surface();
  } else {
    vulkan_device_ = static_cast<taichi::lang::vulkan::VulkanDevice *>(
        prog->get_graphics_device());
    native_surface_ = make_vk_surface(vulkan_device_->vk_instance());
  }
}

taichi::lang::vulkan::VulkanDevice &AppContext::device() {
  if (vulkan_device_) {
    return *vulkan_device_;
  }
  return *(embedded_vulkan_device_->device());
}

const taichi::lang::vulkan::VulkanDevice &AppContext::device() const {
  if (vulkan_device_) {
    return *vulkan_device_;
  }
  return *(embedded_vulkan_device_->device());
}

AppContext::~AppContext() {
}

bool AppContext::requires_export_sharing() const {
  // only the cuda backends needs export_sharing to interop with vk
  // with other backends (e.g. vulkan backend on mac), turning export_sharing to
  // true leads to crashes
  // TODO: investigate this, and think of a more universal solution.
  return config.ti_arch == Arch::cuda;
}

Pipeline *AppContext::get_raster_pipeline(const RasterPipelineConfig &config) {
  const std::string key = fmt::format(
      "{}${}${}${}${}${}${}", int(config.polygon_mode), int(config.blend),
      config.frag_path, config.vert_path, int(config.prim_topology),
      int(config.depth), int(config.vbo_instanced));
  const auto &iter = pipelines_.find(key);
  if (iter != pipelines_.end()) {
    return iter->second.get();
  } else {
    auto vert_code = read_file(config.vert_path);
    auto frag_code = read_file(config.frag_path);

    std::vector<PipelineSourceDesc> source(2);
    source[0] = {PipelineSourceType::spirv_binary, frag_code.data(),
                 frag_code.size(), PipelineStageType::fragment};
    source[1] = {PipelineSourceType::spirv_binary, vert_code.data(),
                 vert_code.size(), PipelineStageType::vertex};

    RasterParams raster_params;
    raster_params.prim_topology = config.prim_topology;
    raster_params.polygon_mode = config.polygon_mode;
    raster_params.depth_test = config.depth;
    raster_params.depth_write = config.depth;

    if (config.blend) {
      raster_params.blending.push_back(BlendingParams());
    }

    const std::vector<VertexInputBinding> vertex_inputs = {
        {/*binding=*/0, sizeof(Vertex),
         /*instance=*/config.vbo_instanced}};
    // TODO: consider using uint8 for colors and normals
    const std::vector<VertexInputAttribute> vertex_attribs = {
        {/*location=*/0, /*binding=*/0,
         /*format=*/BufferFormat::rgb32f,
         /*offset=*/offsetof(Vertex, pos)},
        {/*location=*/1, /*binding=*/0,
         /*format=*/BufferFormat::rgb32f,
         /*offset=*/offsetof(Vertex, normal)},
        {/*location=*/2, /*binding=*/0,
         /*format=*/BufferFormat::rg32f,
         /*offset=*/offsetof(Vertex, tex_coord)},
        {/*location=*/3, /*binding=*/0,
         /*format=*/BufferFormat::rgba32f,
         /*offset=*/offsetof(Vertex, color)},
    };

    auto pipeline = device().create_raster_pipeline(
        source, raster_params, vertex_inputs, vertex_attribs);

    Pipeline *pp = pipeline.get();
    pipelines_[key] = std::move(pipeline);
    return pp;
  }
}

taichi::lang::Pipeline *AppContext::get_customized_raster_pipeline(
    const RasterPipelineConfig &config,
    const std::vector<taichi::lang::VertexInputBinding> &vertex_inputs,
    const std::vector<taichi::lang::VertexInputAttribute> &vertex_attribs) {
  const std::string key =
      fmt::format("{}${}${}${}${}${}$C", int(config.polygon_mode),
                  int(config.blend), config.frag_path, config.vert_path,
                  int(config.prim_topology), int(config.depth));
  const auto &iter = pipelines_.find(key);
  if (iter != pipelines_.end()) {
    return iter->second.get();
  } else {
    auto vert_code = read_file(config.vert_path);
    auto frag_code = read_file(config.frag_path);

    std::vector<PipelineSourceDesc> source(2);
    source[0] = {PipelineSourceType::spirv_binary, frag_code.data(),
                 frag_code.size(), PipelineStageType::fragment};
    source[1] = {PipelineSourceType::spirv_binary, vert_code.data(),
                 vert_code.size(), PipelineStageType::vertex};

    RasterParams raster_params;
    raster_params.prim_topology = config.prim_topology;
    raster_params.polygon_mode = config.polygon_mode;
    raster_params.depth_test = config.depth;
    raster_params.depth_write = config.depth;

    if (config.blend) {
      raster_params.blending.push_back(BlendingParams());
    }

    auto pipeline = device().create_raster_pipeline(
        source, raster_params, vertex_inputs, vertex_attribs);

    Pipeline *pp = pipeline.get();
    pipelines_[key] = std::move(pipeline);
    return pp;
  }
}

taichi::lang::Pipeline *AppContext::get_compute_pipeline(
    const std::string &shader_path) {
  const std::string &key = shader_path;
  const auto &iter = pipelines_.find(key);
  if (iter != pipelines_.end()) {
    return iter->second.get();
  } else {
    auto comp_code = read_file(shader_path);
    auto [pipeline, res] = device().create_pipeline_unique(
        {PipelineSourceType::spirv_binary, comp_code.data(), comp_code.size(),
         PipelineStageType::compute});
    TI_ASSERT(res == RhiResult::success);
    Pipeline *pp = pipeline.get();
    pipelines_[key] = std::move(pipeline);
    return pp;
  }
}

TaichiWindow *AppContext::taichi_window() const {
  return taichi_window_;
}

lang::Program *AppContext::prog() const {
  return prog_;
}

}  // namespace vulkan

}  // namespace taichi::ui
