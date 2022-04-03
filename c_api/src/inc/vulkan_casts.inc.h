namespace tvk = ::taichi::lang::vulkan;

tvk::VulkanDeviceCreator *cppcast(EmbeddedVulkanDevice *ptr) {
  return reinterpret_cast<tvk::VulkanDeviceCreator *>(ptr);
}

tvk::VkRuntime *cppcast(VulkanRuntime *ptr) {
  return reinterpret_cast<tvk::VkRuntime *>(ptr);
}
