#pragma once

#include <vector>
#include <set>
#include <memory>
#include <string>

#include "taichi/common/core.h"
#include "taichi/rhi/llvm/llvm_device.h"

namespace taichi {
namespace lang {

namespace flagos {

/**
 * @brief FlagOS Device abstraction for Taichi
 *
 * FlagOS is a unified AI system software stack that provides a common interface
 * for various AI chips. This device implementation bridges Taichi's LLVM
 * backend with FlagOS's FlagTree compiler.
 */

class FlagosPipeline : public Pipeline {
 public:
  ~FlagosPipeline() override {
  }

  // Compiled kernel handle from FlagTree
  void *kernel_handle_{nullptr};
  std::string kernel_name_;
};

class FlagosCommandList : public CommandList {
 public:
  ~FlagosCommandList() override {
  }

  void bind_pipeline(Pipeline *p) noexcept override {
    TI_NOT_IMPLEMENTED;
  }
  RhiResult bind_shader_resources(ShaderResourceSet *res,
                                  int set_index = 0) noexcept override{
      TI_NOT_IMPLEMENTED} RhiResult
      bind_raster_resources(RasterResources *res) noexcept override {
    TI_NOT_IMPLEMENTED
  }
  void buffer_barrier(DevicePtr ptr, size_t size) noexcept override {
    TI_NOT_IMPLEMENTED
  }
  void buffer_barrier(DeviceAllocation alloc) noexcept override {
    TI_NOT_IMPLEMENTED
  }
  void memory_barrier() noexcept override {
    TI_NOT_IMPLEMENTED;
  }
  void buffer_copy(DevicePtr dst,
                   DevicePtr src,
                   size_t size) noexcept override {
    TI_NOT_IMPLEMENTED
  }
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) noexcept override{
      TI_NOT_IMPLEMENTED} RhiResult
      dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) noexcept override {
    TI_NOT_IMPLEMENTED
  }
};

class FlagosStream : public Stream {
 public:
  ~FlagosStream() override {
  }

  RhiResult new_command_list(CommandList **out_cmdlist) noexcept override{
      TI_NOT_IMPLEMENTED} StreamSemaphore
      submit(CommandList *cmdlist,
             const std::vector<StreamSemaphore> &wait_semaphores = {}) override{
          TI_NOT_IMPLEMENTED} StreamSemaphore
      submit_synced(
          CommandList *cmdlist,
          const std::vector<StreamSemaphore> &wait_semaphores = {}) override {
    TI_NOT_IMPLEMENTED
  }

  void command_sync() override {
    TI_NOT_IMPLEMENTED;
  }
};

class FlagosDevice : public LlvmDevice {
 public:
  struct AllocInfo {
    void *ptr{nullptr};
    size_t size{0};
    bool is_imported{false};
    bool use_preallocated{true};
    bool use_cached{false};
    void *mapped{nullptr};
  };

  AllocInfo get_alloc_info(const DeviceAllocation handle);

  FlagosDevice();
  ~FlagosDevice() override;

  // Device capabilities
  Arch arch() const override {
    return Arch::flagos;
  }

  // Memory management
  RhiResult allocate_memory(const AllocParams &params,
                            DeviceAllocation *out_devalloc) override;
  DeviceAllocation allocate_memory_runtime(
      const LlvmRuntimeAllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;

  uint64_t *allocate_llvm_runtime_memory_jit(
      const LlvmRuntimeAllocParams &params) override;

  RhiResult upload_data(DevicePtr *device_ptr,
                        const void **data,
                        size_t *size,
                        int num_alloc = 1) noexcept override;

  RhiResult readback_data(
      DevicePtr *device_ptr,
      void **data,
      size_t *size,
      int num_alloc = 1,
      const std::vector<StreamSemaphore> &wait_sema = {}) noexcept override;

  ShaderResourceSet *create_resource_set() final {
    TI_NOT_IMPLEMENTED;
  }

  RhiResult create_pipeline(Pipeline **out_pipeline,
                            const PipelineSourceDesc &src,
                            std::string name,
                            PipelineCache *cache) noexcept final {
    TI_NOT_IMPLEMENTED;
  }

  RhiResult map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) final;
  RhiResult map(DeviceAllocation alloc, void **mapped_ptr) final;

  void unmap(DevicePtr ptr) override {
    TI_NOT_IMPLEMENTED;
  }
  void unmap(DeviceAllocation alloc) override;

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  DeviceAllocation import_memory(void *ptr, size_t size) override;

  // LLVM Device interface
  void *get_memory_addr(DeviceAllocation devalloc) override {
    return get_alloc_info(devalloc).ptr;
  }

  std::size_t get_total_memory() override;

  Stream *get_compute_stream() override {
    TI_NOT_IMPLEMENTED;
  }

  void wait_idle() override;

  void clear() override;

  // FlagOS specific methods
  /**
   * @brief Initialize FlagOS backend with specific chip target
   * @param chip_name Target chip name (e.g., "mlu370", "ascend910", "dcu")
   */
  void initialize(const std::string &chip_name);

  /**
   * @brief Get the target chip name
   */
  std::string get_target_chip() const {
    return target_chip_;
  }

  /**
   * @brief Check if a specific chip is supported
   */
  static bool is_chip_supported(const std::string &chip_name);

  /**
   * @brief Get list of supported chips
   */
  static std::vector<std::string> get_supported_chips();

  /**
   * @brief Launch a compiled kernel
   */
  void launch_kernel(const std::string &kernel_name,
                     void **args,
                     uint32_t grid_dim,
                     uint32_t block_dim);

 private:
  std::vector<AllocInfo> allocations_;
  std::string target_chip_{"generic"};

  // FlagOS context
  struct FlagosContextWrapper;
  std::unique_ptr<FlagosContextWrapper> context_;

  void validate_device_alloc(const DeviceAllocation alloc) {
    if (allocations_.size() <= alloc.alloc_id) {
      TI_ERROR("invalid DeviceAllocation");
    }
  }
};

}  // namespace flagos

}  // namespace lang

}  // namespace taichi
