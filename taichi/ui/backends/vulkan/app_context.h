#pragma once
#include "taichi/ui/common/app_config.h"
#include <memory>
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
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
  void init(lang::Program *prog, TaichiWindow *window, const AppConfig &config);
  ~AppContext();

  TaichiWindow *taichi_window() const;
  lang::Program *prog() const;

  taichi::lang::vulkan::VulkanDevice &device();
  const taichi::lang::vulkan::VulkanDevice &device() const;
  bool requires_export_sharing() const;

  AppConfig config;

  taichi::lang::Pipeline *get_raster_pipeline(
      const std::string &frag_path,
      const std::string &vert_path,
      taichi::lang::TopologyType prim_topology,
      bool depth = false,
      taichi::lang::PolygonMode polygon_mode = taichi::lang::PolygonMode::Fill,
      bool blend = true,
      bool vbo_instanced = false);

  taichi::lang::Pipeline *get_customized_raster_pipeline(
      const std::string &frag_path,
      const std::string &vert_path,
      taichi::lang::TopologyType prim_topology,
      bool depth,
      taichi::lang::PolygonMode polygon_mode,
      bool blend,
      const std::vector<taichi::lang::VertexInputBinding> &vertex_inputs,
      const std::vector<taichi::lang::VertexInputAttribute> &vertex_attribs);

  taichi::lang::Pipeline *get_compute_pipeline(const std::string &shader_path);

 private:
  std::unique_ptr<taichi::lang::vulkan::VulkanDeviceCreator>
      embedded_vulkan_device_{nullptr};

  // not owned
  taichi::lang::vulkan::VulkanDevice *vulkan_device_{nullptr};

  std::unordered_map<std::string, taichi::lang::UPipeline> pipelines_;

  TaichiWindow *taichi_window_{nullptr};

  lang::Program *prog_{nullptr};
};

}  // namespace vulkan

}  // namespace taichi::ui
