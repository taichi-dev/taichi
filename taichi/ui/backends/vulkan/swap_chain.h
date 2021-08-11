#pragma once
TI_UI_NAMESPACE_BEGIN
namespace vulkan {

struct SwapChain {
  static const int MAX_FRAMES_IN_FLIGHT = 4;

  uint32_t curr_image_index;

  VkSurfaceKHR surface;

  VkSwapchainKHR swap_chain;
  std::vector<VkImage> swap_chain_images;
  VkFormat swap_chain_image_format;
  VkExtent2D swap_chain_extent;
  std::vector<VkImageView> swap_chain_image_views;
  std::vector<VkFramebuffer> swap_chain_framebuffers;

  VkImage depth_image;
  VkDeviceMemory depth_image_memory;
  VkImageView depth_image_view;

  std::vector<VkSemaphore> image_available_semaphores;
  std::vector<VkSemaphore> render_finished_semaphores;
  std::vector<VkFence> in_flight_scenes;
  std::vector<VkFence> images_in_flight;
  size_t current_frame = 0;

  bool requires_recreate = false;

  class AppContext *app_context;

  void update_image_index();

  void cleanup_swap_chain();
  void cleanup();

  void recreate_swap_chain();

  void create_swap_chain();

  void create_image_views();

  void create_framebuffers();

  void create_depth_resources();

  void create_sync_objects();

  uint32_t get_image_index();

  void present_frame();

  VkSurfaceFormatKHR choose_swap_surface_format(
      const std::vector<VkSurfaceFormatKHR> &available_formats);
  VkPresentModeKHR choose_swap_present_mode(
      const std::vector<VkPresentModeKHR> &available_present_modes);
  VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR &capabilities);
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
