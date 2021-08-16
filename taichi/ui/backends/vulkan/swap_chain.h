#pragma once
TI_UI_NAMESPACE_BEGIN
namespace vulkan {

class SwapChain {
 public:
  void init(class AppContext *app_context, VkSurfaceKHR surface);
  void update_image_index();
  void cleanup_swap_chain();
  void cleanup();
  void recreate_swap_chain();

  void create_swap_chain();

  void create_image_views();

  void create_framebuffers();

  void create_depth_resources();

  void create_sync_objects();

  void present_frame();

  bool requires_recreate() const;
  uint32_t curr_image_index();

  uint32_t current_frame() const;

  size_t chain_size() const;

  std::vector<VkFence> &in_flight_scenes();
  std::vector<VkFence> &images_in_flight();
  const std::vector<VkSemaphore> &image_available_semaphores() const;
  const std::vector<VkSemaphore> &render_finished_semaphores() const;

  VkExtent2D swap_chain_extent() const;
  VkFormat swap_chain_image_format() const;
  const std::vector<VkFramebuffer> &swap_chain_framebuffers() const;

  VkSurfaceFormatKHR choose_swap_surface_format(
      const std::vector<VkSurfaceFormatKHR> &available_formats);
  VkPresentModeKHR choose_swap_present_mode(
      const std::vector<VkPresentModeKHR> &available_present_modes);
  VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR &capabilities);

  static const int MAX_FRAMES_IN_FLIGHT = 4;

 private:
  uint32_t curr_image_index_;
  VkSurfaceKHR surface_;
  VkSwapchainKHR swap_chain_;

  std::vector<VkImage> swap_chain_images_;
  VkFormat swap_chain_image_format_;
  VkExtent2D swap_chain_extent_;
  std::vector<VkImageView> swap_chain_image_views_;
  std::vector<VkFramebuffer> swap_chain_framebuffers_;

  VkImage depth_image_;
  VkDeviceMemory depth_image_memory_;
  VkImageView depth_image_view_;

  std::vector<VkSemaphore> image_available_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> in_flight_scenes_;
  std::vector<VkFence> images_in_flight_;
  uint32_t current_frame_ = 0;

  bool requires_recreate_{false};

  class AppContext *app_context_;
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
