namespace tvk = ::taichi::lang::vulkan;

tvk::VulkanDeviceCreator *cppcast(Taichi_EmbeddedVulkanDevice *ptr) {
  return reinterpret_cast<tvk::VulkanDeviceCreator *>(ptr);
}

tvk::VkRuntime *cppcast(Taichi_VulkanRuntime *ptr) {
  return reinterpret_cast<tvk::VkRuntime *>(ptr);
}

tvk::VulkanDevice *cppcast(Taichi_VulkanDevice *dev) {
  return reinterpret_cast<tvk::VulkanDevice *>(dev);
}
