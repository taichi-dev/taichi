namespace tvk = ::taichi::lang::vulkan;


tvk::VkRuntime *cppcast(Taichi_VulkanRuntime *ptr) {
  return reinterpret_cast<tvk::VkRuntime *>(ptr);
}

