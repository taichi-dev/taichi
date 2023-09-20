#pragma once
#include "taichi/ui/common/app_config.h"
#include <memory>
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#ifdef TI_WITH_METAL
#include "taichi/rhi/metal/metal_device.h"
#endif
#include "taichi/ui/ggui/swap_chain.h"
#ifdef ANDROID
#include <android/native_window.h>
#endif

namespace taichi::lang {
class Program;
}  // namespace taichi::lang

namespace taichi::ui {

#ifdef ANDROID
using TaichiWindow = ANativeWindow;
#else
using TaichiWindow = GLFWwindow;
#endif

namespace vulkan {

class TI_DLL_EXPORT AppContext {
 public:
  void init_with_vulkan(lang::Program *prog,
                        TaichiWindow *window,
                        const AppConfig &config);
  void init_with_metal(lang::Program *prog,
                       TaichiWindow *window,
                       const AppConfig &config);
  ~AppContext();

  TaichiWindow *taichi_window() const;
  lang::Program *prog() const;

  taichi::lang::GraphicsDevice &device();
  const taichi::lang::GraphicsDevice &device() const;
  bool requires_export_sharing() const;

  AppConfig config;

  struct RasterPipelineConfig {
    std::string frag_path;
    std::string vert_path;
    taichi::lang::TopologyType prim_topology{
        taichi::lang::TopologyType::Triangles};
    bool depth{false};
    taichi::lang::PolygonMode polygon_mode{taichi::lang::PolygonMode::Fill};
    bool blend{true};
    bool vbo_instanced{false};
  };

  // Get a raster pipeline with the given fragment shader and vertex shader &
  // options.
  // - This function will cache the pipeline for future use.
  // - This function will use the default GGUI vertex input format
  taichi::lang::Pipeline *get_raster_pipeline(
      const RasterPipelineConfig &config);

  // Get a raster pipeline with the given fragment shader and vertex shader &
  // options.
  // - This function will cache the pipeline for future use
  // - This function will use the provided vertex input format
  taichi::lang::Pipeline *get_customized_raster_pipeline(
      const RasterPipelineConfig &config,
      const std::vector<taichi::lang::VertexInputBinding> &vertex_inputs,
      const std::vector<taichi::lang::VertexInputAttribute> &vertex_attribs);

  // Get a compute pipeline with the given compute shader
  // - This function will cache the pipeline for future use
  taichi::lang::Pipeline *get_compute_pipeline(const std::string &shader_path);

  VkSurfaceKHR get_native_surface() const {
    return native_surface_;
  }

 private:
  std::unique_ptr<taichi::lang::vulkan::VulkanDeviceCreator>
      embedded_vulkan_device_{nullptr};

  VkSurfaceKHR native_surface_{VK_NULL_HANDLE};

  std::unordered_map<std::string, taichi::lang::UPipeline> pipelines_;

  // not owned
  taichi::lang::GraphicsDevice *graphics_device_{nullptr};

  TaichiWindow *taichi_window_{nullptr};

  lang::Program *prog_{nullptr};
};

}  // namespace vulkan

}  // namespace taichi::ui
