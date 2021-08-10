#pragma once

#include <volk.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <functional>

// #define TI_VULKAN_DEBUG

#ifdef TI_VULKAN_DEBUG
#include <GLFW/glfw3.h>
#endif

namespace taichi {
namespace lang {
namespace vulkan {

struct SpirvCodeView {
  const uint32_t *data = nullptr;
  size_t size = 0;

  SpirvCodeView() = default;

  explicit SpirvCodeView(const std::vector<uint32_t> &code)
      : data(code.data()), size(code.size() * sizeof(uint32_t)) {
  }
};

struct VulkanQueueFamilyIndices {
  std::optional<uint32_t> compute_family;
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;
  // TODO: While it is the case that all COMPUTE/GRAPHICS queue also support
  // TRANSFER by default, maye there are some performance benefits to find a
  // TRANSFER-dedicated queue family.
  // https://vulkan-tutorial.com/Vertex_buffers/Staging_buffer#page_Transfer-queue

  bool is_complete() const {
    return compute_family.has_value();
  }

  bool is_complete_for_ui() {
    return graphics_family.has_value() && present_family.has_value();
  }
};

#ifdef TI_VULKAN_DEBUG
struct VulkanDeviceDebugStruct {
  GLFWwindow *window{nullptr};
  VkSurfaceKHR surface;
  VkSwapchainKHR swapchain;
  VkSemaphore image_available;
};
#endif

// Many classes here are inspired by TVM's runtime
// https://github.com/apache/tvm/tree/main/src/runtime/vulkan
//
// VulkanDevice maps to a (VkDevice, VkQueue) tuple. Right now we only use
// a single queue from a single device, so it does not make a difference to
// separate the queue from the device. This is similar to using a single CUDA
// stream.
//
// Note that this class does NOT own the underlying Vk* resources. The idea is
// that users of this lib can provide such resources already created in their
// Vulkan pipeline.
//
// TODO: Think of a better class name.
class VulkanDevice {
 public:
  struct Params {
    VkDevice device{VK_NULL_HANDLE};
    VkQueue compute_queue{VK_NULL_HANDLE};
    VkQueue graphics_queue{VK_NULL_HANDLE};
    VkQueue present_queue{VK_NULL_HANDLE};
    VkCommandPool command_pool{VK_NULL_HANDLE};
  };

  explicit VulkanDevice(const Params &params);

  VkDevice device() const {
    return rep_.device;
  }

  VkQueue compute_queue() const {
    return rep_.compute_queue;
  }

  VkQueue graphics_queue() const {
    return rep_.graphics_queue;
  }

  VkQueue present_queue() const {
    return rep_.present_queue;
  }

  VkCommandPool command_pool() const {
    return rep_.command_pool;
  }

#ifdef TI_VULKAN_DEBUG
  void set_debug_struct(VulkanDeviceDebugStruct *s) {
    this->debug_struct_ = s;
  }
#endif

  void debug_frame_marker() const;

 private:
#ifdef TI_VULKAN_DEBUG
  VulkanDeviceDebugStruct *debug_struct_{nullptr};
#endif
  Params rep_;
};

struct VulkanCapabilities {
  uint32_t api_version;
  uint32_t spirv_version;

  bool has_int8{false};
  bool has_int16{false};
  bool has_int64{false};
  bool has_float64{false};

  bool has_nvidia_interop{false};
  bool has_atomic_i64{false};
  bool has_atomic_float{false};
  bool has_presentation{false};
  bool has_spv_variable_ptr{false};
};

/**
 * This class creates a VulkanDevice instance. The underlying Vk* resources are
 * embedded directly inside the class.
 */
class EmbeddedVulkanDevice {
 public:
  struct Params {
    std::optional<uint32_t> api_version;
    bool is_for_ui{false};
    std::vector<const char *> additional_instance_extensions;
    std::vector<const char *> additional_device_extensions;
    // the VkSurfaceKHR needs to be created after creating the VkInstance, but
    // before creating the VkPhysicalDevice thus, we allow the user to pass in a
    // custom surface creator
    std::function<VkSurfaceKHR(VkInstance)> surface_creator;
  };

  explicit EmbeddedVulkanDevice(const Params &params);
  ~EmbeddedVulkanDevice();

  VkInstance instance() {
    return instance_;
  }

  VulkanDevice *device() {
    return owned_device_.get();
  }

  const VulkanDevice *device() const {
    return owned_device_.get();
  }

  VkPhysicalDevice physical_device() const {
    return physical_device_;
  }

  VkSurfaceKHR surface() const {
    return surface_;
  }

  VkInstance instance() const {
    return instance_;
  }

  const VulkanQueueFamilyIndices &queue_family_indices() const {
    return queue_family_indices_;
  }

  const VulkanCapabilities &get_capabilities() const {
    return capability_;
  }

 private:
  void create_instance();
  void setup_debug_messenger();
  void create_surface();
  void pick_physical_device();
  void create_logical_device();
  void create_command_pool();
  void create_debug_swapchain();

#ifdef TI_VULKAN_DEBUG
  VulkanDeviceDebugStruct debug_struct_;
#endif

  VkInstance instance_{VK_NULL_HANDLE};
  VkDebugUtilsMessengerEXT debug_messenger_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VulkanQueueFamilyIndices queue_family_indices_;
  VkDevice device_{VK_NULL_HANDLE};
  // TODO: It's probably not right to put these per-queue things here. However,
  // in Taichi we only use a single queue on a single device (i.e. a single CUDA
  // stream), so it doesn't make a difference.
  VkQueue compute_queue_{VK_NULL_HANDLE};
  VkQueue graphics_queue_{VK_NULL_HANDLE};
  VkQueue present_queue_{VK_NULL_HANDLE};

  VkSurfaceKHR surface_{VK_NULL_HANDLE};

  // TODO: Shall we have dedicated command pools for COMPUTE and TRANSFER
  // commands, respectively?
  VkCommandPool command_pool_{VK_NULL_HANDLE};

  VulkanCapabilities capability_;

  std::unique_ptr<VulkanDevice> owned_device_{nullptr};

  Params params_;
};

// VulkanPipeline maps to a VkPipeline, or a SPIR-V module (a GLSL compute
// shader). Because Taichi's buffers are all pre-allocated upon startup, we
// only need to set up the descriptor set (i.e., bind the buffers via
// VkWriteDescriptorSet) once during the pipeline initialization.
class VulkanPipeline {
 public:
  struct BufferBinding {
    VkBuffer buffer{VK_NULL_HANDLE};
    uint32_t binding{0};
  };

  struct Params {
    const VulkanDevice *device{nullptr};
    std::vector<BufferBinding> buffer_bindings;
    SpirvCodeView code;
    std::string name{"Pipeline"};
  };

  explicit VulkanPipeline(const Params &params);
  ~VulkanPipeline();

  VkPipelineLayout pipeline_layout() const {
    return pipeline_layout_;
  }
  VkPipeline pipeline() const {
    return pipeline_;
  }
  const VkDescriptorSet &descriptor_set() const {
    return descriptor_set_;
  }
  const std::string &name() const {
    return name_;
  }

 private:
  void create_descriptor_set_layout(const Params &params);
  void create_compute_pipeline(const Params &params);
  void create_descriptor_pool(const Params &params);
  void create_descriptor_sets(const Params &params);

  VkDevice device_{VK_NULL_HANDLE};  // not owned

  std::string name_;

  // TODO: Commands using the same Taichi buffers should be able to share the
  // same descriptor set layout?
  VkDescriptorSetLayout descriptor_set_layout_{VK_NULL_HANDLE};
  // TODO: Commands having the same |descriptor_set_layout_| should be able to
  // share the same pipeline layout?
  VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
  // This maps 1:1 to a shader, so it needs to be created per compute
  // shader.
  VkPipeline pipeline_{VK_NULL_HANDLE};
  VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
  VkDescriptorSet descriptor_set_{VK_NULL_HANDLE};
};

enum class VulkanCopyBufferDirection {
  H2D,
  D2H,
  // D2D does not have a use case yet
};

// VulkanCommandBuilder builds a VkCommandBuffer by recording a given series of
// VulkanPipelines. The workgroup count needs to be known at recording time.
// TODO: Do we ever need to adjust the workgroup count at runtime?
class VulkanCommandBuilder {
 public:
  explicit VulkanCommandBuilder(const VulkanDevice *device);

  ~VulkanCommandBuilder();

  VkCommandBuffer build();

  void dispatch(const VulkanPipeline &pipeline, int group_count_x);

  void copy(VkBuffer src_buffer,
            VkBuffer dst_buffer,
            VkDeviceSize size,
            VulkanCopyBufferDirection direction);

 protected:
  // VkCommandBuffers are destroyed when the underlying command pool is
  // destroyed.
  // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers#page_Command-buffer-allocation
  VkCommandBuffer command_buffer_{VK_NULL_HANDLE};
  VkDevice device_;  // do not own
};

VkCommandBuffer record_copy_buffer_command(const VulkanDevice *device,
                                           VkBuffer src_buffer,
                                           VkBuffer dst_buffer,
                                           VkDeviceSize size,
                                           VulkanCopyBufferDirection direction);

// A vulkan stream models an asynchronous GPU execution queue.
// Commands are submitted via launch() and executed asynchronously.
// synchronize()s blocks the host, until all the launched commands have
// completed execution.
class VulkanStream {
 public:
  VulkanStream(const VulkanDevice *device);

  void launch(VkCommandBuffer command);
  void synchronize();

  const VulkanDevice *device() const {
    return device_;
  }

 private:
  const VulkanDevice *const device_;

  std::vector<VkFence> in_flight_fences_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
