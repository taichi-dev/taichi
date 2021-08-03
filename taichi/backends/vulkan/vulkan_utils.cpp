#include "taichi/backends/vulkan/vulkan_utils.h"

#include <spirv-tools/libspirv.hpp>

namespace taichi {
namespace lang {
namespace vulkan {
namespace {

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

}  // namespace

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
