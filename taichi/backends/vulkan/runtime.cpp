#include "taichi/backends/vulkan/runtime.h"

#include <chrono>
#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "taichi/util/environ_config.h"

#ifdef TI_WITH_VULKAN
#include <volk.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

#include "taichi/backends/vulkan/vulkan_api.h"
#include "taichi/backends/vulkan/vulkan_common.h"
#include "taichi/backends/vulkan/vulkan_memory.h"
#include "taichi/backends/vulkan/vulkan_utils.h"

#include "vk_mem_alloc.h"
#endif  // TI_WITH_VULKAN

#include "taichi/math/arithmetic.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {
namespace vulkan {

#ifdef TI_WITH_VULKAN

namespace {
class StopWatch {
 public:
  StopWatch() : begin_(std::chrono::system_clock::now()) {
  }

  int GetMicros() {
    typedef std::chrono::duration<float> fsec;

    auto now = std::chrono::system_clock::now();

    fsec fs = now - begin_;
    begin_ = now;
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(fs);
    return d.count();
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> begin_;
};

using BufferEnum = TaskAttributes::Buffers;
using InputBuffersMap = std::unordered_map<BufferEnum, VkBufferWithMemory *>;

class HostDeviceContextBlitter {
 public:
  HostDeviceContextBlitter(const KernelContextAttributes *ctx_attribs,
                           Context *host_ctx,
                           const VulkanCapabilities *capabilities,
                           uint64_t *host_result_buffer,
                           VkBufferWithMemory *device_buffer,
                           VkBufferWithMemory *host_shadow_buffer)
      : ctx_attribs_(ctx_attribs),
        host_ctx_(host_ctx),
        capabilities_(capabilities),
        host_result_buffer_(host_result_buffer),
        device_buffer_(device_buffer),
        host_shadow_buffer_(host_shadow_buffer) {
  }

  void host_to_device() {
    if (ctx_attribs_->empty()) {
      return;
    }
    auto mapped = device_buffer_->map_mem();
    char *const device_base = reinterpret_cast<char *>(mapped.data());

#define TO_DEVICE(short_type, type)                    \
  if (dt->is_primitive(PrimitiveTypeID::short_type)) { \
    auto d = host_ctx_->get_arg<type>(i);              \
    std::memcpy(device_ptr, &d, sizeof(d));            \
    break;                                             \
  }

    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      const auto dt = arg.dt;
      char *device_ptr = device_base + arg.offset_in_mem;
      do {
        if (arg.is_array) {
          const void *host_ptr = host_ctx_->get_arg<void *>(i);
          std::memcpy(device_ptr, host_ptr, arg.stride);
          break;
        }
        if (capabilities_->has_int8) {
          TO_DEVICE(i8, int8)
          TO_DEVICE(u8, uint8)
        }
        if (capabilities_->has_int16) {
          TO_DEVICE(i16, int16)
          TO_DEVICE(u16, uint16)
        }
        TO_DEVICE(i32, int32)
        TO_DEVICE(u32, uint32)
        TO_DEVICE(f32, float32)
        if (capabilities_->has_int64) {
          TO_DEVICE(i64, int64)
          TO_DEVICE(u64, uint64)
        }
        if (capabilities_->has_float64) {
          TO_DEVICE(f64, float64)
        }
        TI_ERROR("Vulkan does not support arg type={}", data_type_name(arg.dt));
      } while (0);
    }
    char *device_ptr = device_base + ctx_attribs_->extra_args_mem_offset();
    std::memcpy(device_ptr, host_ctx_->extra_args,
                ctx_attribs_->extra_args_bytes());
#undef TO_DEVICE
  }

  void device_to_host() {
    if (ctx_attribs_->empty()) {
      return;
    }
    auto mapped = host_shadow_buffer_->map_mem();
    char *const device_base = reinterpret_cast<char *>(mapped.data());

    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      char *device_ptr = device_base + arg.offset_in_mem;
      if (arg.is_array) {
        void *host_ptr = host_ctx_->get_arg<void *>(i);
        std::memcpy(host_ptr, device_ptr, arg.stride);
      }
    }

#define TO_HOST(short_type, type)                          \
  if (dt->is_primitive(PrimitiveTypeID::short_type)) {     \
    const type d = *reinterpret_cast<type *>(device_ptr);  \
    host_result_buffer_[i] =                               \
        taichi_union_cast_with_different_sizes<uint64>(d); \
    break;                                                 \
  }

    for (int i = 0; i < ctx_attribs_->rets().size(); ++i) {
      // Note that we are copying the i-th return value on Metal to the i-th
      // *arg* on the host context.
      const auto &ret = ctx_attribs_->rets()[i];
      char *device_ptr = device_base + ret.offset_in_mem;
      const auto dt = ret.dt;
      do {
        if (ret.is_array) {
          void *host_ptr = host_ctx_->get_arg<void *>(i);
          std::memcpy(host_ptr, device_ptr, ret.stride);
          break;
        }
        if (capabilities_->has_int8) {
          TO_HOST(i8, int8)
          TO_HOST(u8, uint8)
        }
        if (capabilities_->has_int16) {
          TO_HOST(i16, int16)
          TO_HOST(u16, uint16)
        }
        TO_HOST(i32, int32)
        TO_HOST(u32, uint32)
        TO_HOST(f32, float32)
        if (capabilities_->has_int64) {
          TO_HOST(i64, int64)
          TO_HOST(u64, uint64)
        }
        if (capabilities_->has_float64) {
          TO_HOST(f64, float64)
        }
        TI_ERROR("Vulkan does not support return value type={}",
                 data_type_name(ret.dt));
      } while (0);
    }
#undef TO_HOST
  }

  static std::unique_ptr<HostDeviceContextBlitter> maybe_make(
      const KernelContextAttributes *ctx_attribs,
      Context *host_ctx,
      const VulkanCapabilities *capabilities,
      uint64_t *host_result_buffer,
      VkBufferWithMemory *device_buffer,
      VkBufferWithMemory *host_shadow_buffer) {
    if (ctx_attribs->empty()) {
      return nullptr;
    }
    return std::make_unique<HostDeviceContextBlitter>(
        ctx_attribs, host_ctx, capabilities, host_result_buffer, device_buffer,
        host_shadow_buffer);
  }

 private:
  const KernelContextAttributes *const ctx_attribs_;
  Context *const host_ctx_;
  const VulkanCapabilities *capabilities_;
  uint64_t *const host_result_buffer_;
  VkBufferWithMemory *const device_buffer_;
  VkBufferWithMemory *const host_shadow_buffer_;
};

// Info for launching a compiled Taichi kernel, which consists of a series of
// Vulkan pipelines.
class CompiledTaichiKernel {
 public:
  struct Params {
    const TaichiKernelAttributes *ti_kernel_attribs{nullptr};
    std::vector<VkRuntime::SpirvBinary> spirv_bins;
    const SNodeDescriptorsMap *snode_descriptors{nullptr};

    const VulkanDevice *device{nullptr};
    VkBufferWithMemory *root_buffer{nullptr};
    VkBufferWithMemory *global_tmps_buffer{nullptr};
    VmaAllocator *allocator{nullptr};
  };

  CompiledTaichiKernel(const Params &ti_params)
      : ti_kernel_attribs_(*ti_params.ti_kernel_attribs) {
    InputBuffersMap input_buffers = {
        {BufferEnum::Root, ti_params.root_buffer},
        {BufferEnum::GlobalTmps, ti_params.global_tmps_buffer},
    };
    const auto ctx_sz = ti_kernel_attribs_.ctx_attribs.total_bytes();
    if (!ti_kernel_attribs_.ctx_attribs.empty()) {
      ctx_buffer_ = std::make_unique<VkBufferWithMemory>(
          *ti_params.allocator, ctx_sz,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          /*host_write*/ true, /*host_read*/ false);
      ctx_buffer_host_ = std::make_unique<VkBufferWithMemory>(
          *ti_params.allocator, ctx_sz,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          /*host_write*/ false, /*host_read*/ true);
      input_buffers[BufferEnum::Context] = ctx_buffer_.get();
    }

    const auto &task_attribs = ti_kernel_attribs_.tasks_attribs;
    const auto &spirv_bins = ti_params.spirv_bins;
    TI_ASSERT(task_attribs.size() == spirv_bins.size());

    VulkanCommandBuilder cmd_builder(ti_params.device);
    for (int i = 0; i < task_attribs.size(); ++i) {
      const auto &attribs = task_attribs[i];
      VulkanPipeline::Params vp_params;
      vp_params.device = ti_params.device;
      vp_params.name = ti_kernel_attribs_.name;
      for (const auto &bb : task_attribs[i].buffer_binds) {
        vp_params.buffer_bindings.push_back(VulkanPipeline::BufferBinding{
            input_buffers.at(bb.type)->buffer(), (uint32_t)bb.binding});
      }
      vp_params.code = SpirvCodeView(spirv_bins[i]);
      auto vp = std::make_unique<VulkanPipeline>(vp_params);
      const int group_x = (attribs.advisory_total_num_threads +
                           attribs.advisory_num_threads_per_group - 1) /
                          attribs.advisory_num_threads_per_group;
      cmd_builder.dispatch(*vp, group_x);
      vk_pipelines_.push_back(std::move(vp));
    }

    if (!ti_kernel_attribs_.ctx_attribs.empty()) {
      cmd_builder.copy(ctx_buffer_->buffer(), ctx_buffer_host_->buffer(),
                       ctx_sz, VulkanCopyBufferDirection::D2H);
    }

    command_buffer_ = cmd_builder.build();
  }

  const TaichiKernelAttributes &ti_kernel_attribs() const {
    return ti_kernel_attribs_;
  }

  size_t num_vk_pipelines() const {
    return vk_pipelines_.size();
  }

  VkBufferWithMemory *ctx_buffer() const {
    return ctx_buffer_.get();
  }

  VkBufferWithMemory *ctx_buffer_host() const {
    return ctx_buffer_host_.get();
  }

  VkCommandBuffer command_buffer() const {
    return command_buffer_;
  }

 private:
  TaichiKernelAttributes ti_kernel_attribs_;

  // Right now |ctx_buffer_| is allocated from a HOST_VISIBLE|COHERENT
  // memory, because we do not do computation on this buffer anyway, and it may
  // not worth the effort doing another hop via a staging buffer.
  // TODO: Provide an option to use staging buffer. This could be useful if the
  // kernel does lots of IO on the context buffer, e.g., copy a large np array.
  std::unique_ptr<VkBufferWithMemory> ctx_buffer_{nullptr};
  std::unique_ptr<VkBufferWithMemory> ctx_buffer_host_{nullptr};
  std::vector<std::unique_ptr<VulkanPipeline>> vk_pipelines_;

  // VkCommandBuffers are destroyed when the underlying command pool is
  // destroyed.
  // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers#page_Command-buffer-allocation
  VkCommandBuffer command_buffer_{VK_NULL_HANDLE};
};

class ClearBufferCommandBuilder : private VulkanCommandBuilder {
 public:
  using VulkanCommandBuilder::VulkanCommandBuilder;

  VkCommandBuffer build(const std::vector<VkBuffer> &buffers) {
    for (auto b : buffers) {
      vkCmdFillBuffer(command_buffer_, b, /*dstOffset=*/0,
                      /*size=*/VK_WHOLE_SIZE,
                      /*data=*/0);
    }
    return VulkanCommandBuilder::build();
  }
};

}  // namespace

class VkRuntime ::Impl {
 private:
  std::unique_ptr<spvtools::SpirvTools> spirv_tools_;
  std::unique_ptr<spvtools::Optimizer> spirv_opt_;

  spvtools::OptimizerOptions _spirv_opt_options;

  static void spriv_message_consumer(spv_message_level_t level,
                                     const char *source,
                                     const spv_position_t &position,
                                     const char *message) {
    // TODO: Maybe we can add a macro, e.g. TI_LOG_AT_LEVEL(lv, ...)
    if (level <= SPV_MSG_FATAL) {
      TI_ERROR("{}\n[{}:{}:{}] {}", source, position.index, position.line,
               position.column, message);
    } else if (level <= SPV_MSG_WARNING) {
      TI_WARN("{}\n[{}:{}:{}] {}", source, position.index, position.line,
              position.column, message);
    } else if (level <= SPV_MSG_INFO) {
      TI_INFO("{}\n[{}:{}:{}] {}", source, position.index, position.line,
              position.column, message);
    } else if (level <= SPV_MSG_INFO) {
      TI_TRACE("{}\n[{}:{}:{}] {}", source, position.index, position.line,
               position.column, message);
    }
  };

 public:
  explicit Impl(const Params &params)
      : snode_descriptors_(params.snode_descriptors),
        host_result_buffer_(params.host_result_buffer) {
    TI_ASSERT(snode_descriptors_ != nullptr);
    TI_ASSERT(host_result_buffer_ != nullptr);
    EmbeddedVulkanDevice::Params evd_params;
    evd_params.api_version = VulkanEnvSettings::kApiVersion();
    embedded_device_ = std::make_unique<EmbeddedVulkanDevice>(evd_params);
    stream_ = std::make_unique<VulkanStream>(embedded_device_->device());

    init_memory_pool(params);
    init_vk_buffers();

    spirv_tools_ = std::make_unique<spvtools::SpirvTools>(SPV_ENV_VULKAN_1_2);
    spirv_opt_ = std::make_unique<spvtools::Optimizer>(SPV_ENV_VULKAN_1_2);

    spirv_tools_->SetMessageConsumer(spriv_message_consumer);
    spirv_opt_->SetMessageConsumer(spriv_message_consumer);

    // FIXME: Utilize this if KHR_memory_model is supported
    // spirv_opt_->RegisterPass(spvtools::CreateUpgradeMemoryModelPass());
    spirv_opt_->RegisterPerformancePasses();

    for (auto &p : spirv_opt_->GetPassNames()) {
      TI_TRACE("SPIRV Optimization Pass {}", p);
    }

    _spirv_opt_options.set_run_validator(false);
  }

  ~Impl() {
    {
      decltype(ti_kernels_) tmp;
      tmp.swap(ti_kernels_);
    }
    global_tmps_buffer_.reset();
    root_buffer_.reset();
    vmaDestroyAllocator(vk_allocator_);
  }

  KernelHandle register_taichi_kernel(RegisterParams reg_params) {
    CompiledTaichiKernel::Params params;
    params.ti_kernel_attribs = &(reg_params.kernel_attribs);
    params.snode_descriptors = snode_descriptors_;
    params.device = embedded_device_->device();
    params.root_buffer = root_buffer_.get();
    params.global_tmps_buffer = global_tmps_buffer_.get();
    params.allocator = &vk_allocator_;

    for (int i = 0; i < reg_params.task_spirv_source_codes.size(); ++i) {
      const auto &attribs = reg_params.kernel_attribs.tasks_attribs[i];
      const auto &spirv_src = reg_params.task_spirv_source_codes[i];
      const auto &task_name = attribs.name;

      TI_WARN_IF(!spirv_tools_->Validate(spirv_src), "SPIRV validation failed");

      std::vector<uint32_t> optimized_spv;

      TI_WARN_IF(!spirv_opt_->Run(spirv_src.data(), spirv_src.size(),
                                  &optimized_spv, _spirv_opt_options),
                 "SPIRV optimization failed");

      TI_TRACE("SPIRV-Tools-opt: binary size, before={}, after={}",
               spirv_src.size(), optimized_spv.size());

      // Enable to dump SPIR-V assembly of kernels
#if 0
      std::string spirv_asm;
      spirv_tools_->Disassemble(optimized_spv, &spirv_asm);
      TI_TRACE("SPIR-V Assembly dump:\n{}\n\n", spirv_asm);
#endif

      // If we can reach here, we have succeeded. Otherwise
      // std::optional::value() would have killed us.
      params.spirv_bins.push_back(std::move(optimized_spv));
    }
    KernelHandle res;
    res.id_ = ti_kernels_.size();
    ti_kernels_.push_back(std::make_unique<CompiledTaichiKernel>(params));
    return res;
  }

  void launch_kernel(KernelHandle handle, Context *host_ctx) {
    auto *ti_kernel = ti_kernels_[handle.id_].get();
    auto ctx_blitter = HostDeviceContextBlitter::maybe_make(
        &ti_kernel->ti_kernel_attribs().ctx_attribs, host_ctx,
        &get_capabilities(), host_result_buffer_, ti_kernel->ctx_buffer(),
        ti_kernel->ctx_buffer_host());
    if (ctx_blitter) {
      TI_ASSERT(ti_kernel->ctx_buffer() != nullptr);
      ctx_blitter->host_to_device();
    }

    stream_->launch(ti_kernel->command_buffer());
    num_pending_kernels_ += ti_kernel->num_vk_pipelines();
    if (ctx_blitter) {
      synchronize();
      ctx_blitter->device_to_host();
    }
  }

  void synchronize() {
    if (num_pending_kernels_ == 0) {
      return;
    }

    stream_->synchronize();
    num_pending_kernels_ = 0;
  }

  const VulkanCapabilities &get_capabilities() const {
    return embedded_device_->get_capabilities();
  }

 private:
  void init_memory_pool(const Params &params) {
    VolkDeviceTable table;
    VmaVulkanFunctions vk_vma_functions;

    volkLoadDeviceTable(&table, embedded_device_->device()->device());
    vk_vma_functions.vkGetPhysicalDeviceProperties =
        PFN_vkGetPhysicalDeviceProperties(vkGetInstanceProcAddr(
            volkGetLoadedInstance(), "vkGetPhysicalDeviceProperties"));
    vk_vma_functions.vkGetPhysicalDeviceMemoryProperties =
        PFN_vkGetPhysicalDeviceMemoryProperties(vkGetInstanceProcAddr(
            volkGetLoadedInstance(), "vkGetPhysicalDeviceMemoryProperties"));
    vk_vma_functions.vkAllocateMemory = table.vkAllocateMemory;
    vk_vma_functions.vkFreeMemory = table.vkFreeMemory;
    vk_vma_functions.vkMapMemory = table.vkMapMemory;
    vk_vma_functions.vkUnmapMemory = table.vkUnmapMemory;
    vk_vma_functions.vkFlushMappedMemoryRanges =
        table.vkFlushMappedMemoryRanges;
    vk_vma_functions.vkInvalidateMappedMemoryRanges =
        table.vkInvalidateMappedMemoryRanges;
    vk_vma_functions.vkBindBufferMemory = table.vkBindBufferMemory;
    vk_vma_functions.vkBindImageMemory = table.vkBindImageMemory;
    vk_vma_functions.vkGetBufferMemoryRequirements =
        table.vkGetBufferMemoryRequirements;
    vk_vma_functions.vkGetImageMemoryRequirements =
        table.vkGetImageMemoryRequirements;
    vk_vma_functions.vkCreateBuffer = table.vkCreateBuffer;
    vk_vma_functions.vkDestroyBuffer = table.vkDestroyBuffer;
    vk_vma_functions.vkCreateImage = table.vkCreateImage;
    vk_vma_functions.vkDestroyImage = table.vkDestroyImage;
    vk_vma_functions.vkCmdCopyBuffer = table.vkCmdCopyBuffer;
    vk_vma_functions.vkGetBufferMemoryRequirements2KHR =
        table.vkGetBufferMemoryRequirements2KHR;
    vk_vma_functions.vkGetImageMemoryRequirements2KHR =
        table.vkGetImageMemoryRequirements2KHR;
    vk_vma_functions.vkBindBufferMemory2KHR = table.vkBindBufferMemory2KHR;
    vk_vma_functions.vkBindImageMemory2KHR = table.vkBindImageMemory2KHR;
    vk_vma_functions.vkGetPhysicalDeviceMemoryProperties2KHR =
        PFN_vkGetPhysicalDeviceMemoryProperties2KHR(
            vkGetInstanceProcAddr(volkGetLoadedInstance(),
                                  "vkGetPhysicalDeviceMemoryProperties2KHR"));

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.vulkanApiVersion = get_capabilities().api_version;
    allocatorInfo.physicalDevice = embedded_device_->physical_device();
    allocatorInfo.device = embedded_device_->device()->device();
    allocatorInfo.instance = embedded_device_->instance();
    allocatorInfo.pVulkanFunctions = &vk_vma_functions;

    vmaCreateAllocator(&allocatorInfo, &vk_allocator_);
  }

  void init_vk_buffers() {
#pragma message("Vulkan buffers size hardcoded")
    root_buffer_ = std::make_unique<VkBufferWithMemory>(
        vk_allocator_, 64 * 1024 * 1024,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    global_tmps_buffer_ = std::make_unique<VkBufferWithMemory>(
        vk_allocator_, 1024 * 1024,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Need to zero fill the buffers, otherwise there could be NaN.
    ClearBufferCommandBuilder cmd_builder{stream_->device()};
    auto clear_cmd = cmd_builder.build(
        /*buffers=*/{root_buffer_->buffer(), global_tmps_buffer_->buffer()});
    stream_->launch(clear_cmd);
    stream_->synchronize();
  }

  const SNodeDescriptorsMap *const snode_descriptors_;
  uint64_t *const host_result_buffer_;

  std::unique_ptr<EmbeddedVulkanDevice> embedded_device_{nullptr};
  std::unique_ptr<VulkanStream> stream_{nullptr};

  std::unique_ptr<VkBufferWithMemory> root_buffer_;
  std::unique_ptr<VkBufferWithMemory> global_tmps_buffer_;

  VmaAllocator vk_allocator_;

  std::vector<std::unique_ptr<CompiledTaichiKernel>> ti_kernels_;
  int num_pending_kernels_{0};
};

#else

class VkRuntime::Impl {
 public:
  Impl(const Params &) {
    TI_ERROR("Vulkan disabled");
  }

  KernelHandle register_taichi_kernel(RegisterParams) {
    TI_ERROR("Vulkan disabled");
    return KernelHandle();
  }

  void launch_kernel(KernelHandle, Context *) {
    TI_ERROR("Vulkan disabled");
  }

  void synchronize() {
    TI_ERROR("Vulkan disabled");
  }
};

#endif  // TI_WITH_VULKAN

VkRuntime::VkRuntime(const Params &params)
    : impl_(std::make_unique<Impl>(params)) {
}

VkRuntime::~VkRuntime() {
}

VkRuntime::KernelHandle VkRuntime::register_taichi_kernel(
    RegisterParams params) {
  return impl_->register_taichi_kernel(std::move(params));
}

void VkRuntime::launch_kernel(KernelHandle handle, Context *host_ctx) {
  impl_->launch_kernel(handle, host_ctx);
}

void VkRuntime::synchronize() {
  impl_->synchronize();
}

#ifdef TI_WITH_VULKAN
const VulkanCapabilities &VkRuntime::get_capabilities() const {
  return impl_->get_capabilities();
}
#endif

bool is_vulkan_api_available() {
#ifdef TI_WITH_VULKAN
  return true;
#else
  return false;
#endif  // TI_WITH_VULKAN
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
