#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <memory>
#include <optional>
#include <vector>

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
  // TODO: While it is the case that all COMPUTE/GRAPHICS queue also support
  // TRANSFER by default, maye there are some performance benefits to find a
  // TRANSFER-dedicated queue family.
  // https://vulkan-tutorial.com/Vertex_buffers/Staging_buffer#page_Transfer-queue

  bool is_complete() const {
    return compute_family.has_value();
  }
};

// Many classes here are inspired by TVM's runtime
// https://github.com/apache/tvm/tree/main/src/runtime/vulkan
//
// VulkanDevice maps to a (VkDevice, VkQueue) tuple. Right now we only use
// a single queue from a single device, so it does not make a difference to
// separate the queue from the device. This is similar to using a single CUDA
// stream.
//
// TODO: Think of a better class name.
class VulkanDevice {
 public:
  struct Params {
    VkDevice device{VK_NULL_HANDLE};
    VkQueue compute_queue{VK_NULL_HANDLE};
    VkCommandPool command_pool{VK_NULL_HANDLE};
  };

  explicit VulkanDevice(const Params &params);

  VkDevice device() const {
    return rep_.device;
  }

  VkQueue compute_queue() const {
    return rep_.compute_queue;
  }

  VkCommandPool command_pool() const {
    return rep_.command_pool;
  }

 private:
  Params rep_;
};

/**
 * Manages a VulkanDevice instance, including its resources.
 */
class ManagedVulkanDevice {
 public:
  struct Params {
    uint32_t api_version{VK_API_VERSION_1_0};
  };

  explicit ManagedVulkanDevice(const Params &params);
  ~ManagedVulkanDevice();

  VulkanDevice *device() {
    return owned_device_.get();
  }

  const VulkanDevice *device() const {
    return owned_device_.get();
  }

  VkPhysicalDevice physical_device() const {
    return physical_device_;
  }

  const VulkanQueueFamilyIndices &queue_family_indices() const {
    return queue_family_indices_;
  }

 private:
  void create_instance(const Params &params);
  void setup_debug_messenger();
  void pick_physical_device();
  void create_logical_device();
  void create_command_pool();

  VkInstance instance_{VK_NULL_HANDLE};
  VkDebugUtilsMessengerEXT debug_messenger_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VulkanQueueFamilyIndices queue_family_indices_;
  VkDevice device_{VK_NULL_HANDLE};
  // TODO: It's probably not right to put these per-queue things here. However,
  // in Taichi we only use a single queue on a single device (i.e. a single CUDA
  // stream), so it doesn't make a difference.
  VkQueue compute_queue_{VK_NULL_HANDLE};
  // TODO: Shall we have dedicated command pools for COMPUTE and TRANSFER
  // commands, respectively?
  VkCommandPool command_pool_{VK_NULL_HANDLE};

  std::unique_ptr<VulkanDevice> owned_device_{nullptr};
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

 private:
  void create_descriptor_set_layout(const Params &params);
  void create_compute_pipeline(const Params &params);
  void create_descriptor_pool(const Params &params);
  void create_descriptor_sets(const Params &params);

  VkDevice device_{VK_NULL_HANDLE};  // not owned

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

// VulkanCommandBuilder builds a VkCommandBuffer by recording a given series of
// VulkanPipelines. The workgroup count needs to be known at recording time.
// TODO: Do we ever need to adjust the workgroup count at runtime?
class VulkanCommandBuilder {
 public:
  explicit VulkanCommandBuilder(const VulkanDevice *device);

  ~VulkanCommandBuilder();

  VkCommandBuffer build();

 protected:
  // VkCommandBuffers are destroyed when the underlying command pool is
  // destroyed.
  // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers#page_Command-buffer-allocation
  VkCommandBuffer command_buffer_{VK_NULL_HANDLE};
};

class VulkanComputeCommandBuilder : public VulkanCommandBuilder {
 public:
  using VulkanCommandBuilder::VulkanCommandBuilder;

  void append(const VulkanPipeline &pipeline, int group_count_x);
};

enum class VulkanCopyBufferDirection {
  H2D,
  D2H,
  // D2D does not have a use case yet
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
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
