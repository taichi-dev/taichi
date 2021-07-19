#include "taichi/backends/vulkan/vulkan_utils.h"

#include <spirv-tools/libspirv.hpp>

namespace taichi {
namespace lang {
namespace vulkan {

std::vector<VkExtensionProperties> GetInstanceExtensionProperties() {
  constexpr char *kNoLayerName = nullptr;
  uint32_t count = 0;
  vkEnumerateInstanceExtensionProperties(kNoLayerName, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateInstanceExtensionProperties(kNoLayerName, &count,
                                         extensions.data());
  return extensions;
}

std::vector<VkExtensionProperties> GetDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice) {
  constexpr char *kNoLayerName = nullptr;
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(physicalDevice, kNoLayerName, &count,
                                       nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateDeviceExtensionProperties(physicalDevice, kNoLayerName, &count,
                                       extensions.data());
  return extensions;
}

GlslToSpirvCompiler::GlslToSpirvCompiler(const ErrorHandler &err_handler)
    : err_handler_(err_handler) {
  opts_.SetTargetEnvironment(shaderc_target_env_vulkan,
                             VulkanEnvSettings::kShadercEnvVersion());
  opts_.SetOptimizationLevel(shaderc_optimization_level_performance);
}

std::optional<GlslToSpirvCompiler::SpirvBinary> GlslToSpirvCompiler::compile(
    const std::string &glsl_src,
    const std::string &shader_name) {
  auto spv_result =
      compiler_.CompileGlslToSpv(glsl_src, shaderc_glsl_default_compute_shader,
                                 /*input_file_name=*/shader_name.c_str(),
                                 /*entry_point_name=*/"main", opts_);
  if (spv_result.GetCompilationStatus() != shaderc_compilation_status_success) {
    err_handler_(glsl_src, shader_name, spv_result.GetErrorMessage());
    return std::nullopt;
  }
  return SpirvBinary(spv_result.begin(), spv_result.end());
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
