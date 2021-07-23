#pragma once

#include <vulkan/vulkan.h>

#include <functional>
#include <optional>
#include <shaderc/shaderc.hpp>
#include <string>
#include <vector>

namespace taichi {
namespace lang {

namespace vulkan {

std::vector<VkExtensionProperties> GetInstanceExtensionProperties();

std::vector<VkExtensionProperties> GetDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice);

class VulkanEnvSettings {
 public:
  static constexpr uint32_t kApiVersion() {
    return VK_API_VERSION_1_0;
  }

  static constexpr shaderc_env_version kShadercEnvVersion() {
    return shaderc_env_version_vulkan_1_0;
  }
};

class GlslToSpirvCompiler {
 public:
  using SpirvBinary = std::vector<uint32_t>;
  using ErrorHandler = std::function<void(const std::string & /*glsl_src*/,
                                          const std::string & /*shader_name*/,
                                          const std::string & /*err_msg*/)>;

  explicit GlslToSpirvCompiler(const ErrorHandler &err_handler);

  std::optional<SpirvBinary> compile(const std::string &glsl_src,
                                     const std::string &shader_name);

 private:
  shaderc::CompileOptions opts_;
  shaderc::Compiler compiler_;
  ErrorHandler err_handler_{nullptr};
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
