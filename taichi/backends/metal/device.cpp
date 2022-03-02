#include "taichi/backends/metal/device.h"

#include "taichi/platform/mac/objc_api.h"
#include "taichi/backends/metal/api.h"
#include "taichi/backends/metal/constants.h"

namespace taichi {
namespace lang {
namespace metal {
namespace {

class ResourceBinderImpl : public ResourceBinder {
 public:
  struct Binding {
    DeviceAllocationId alloc_id{0};
    // Not sure if this info is necessary yet.
    // TODO: Make it an enum?
    [[maybe_unused]] bool is_constant{false};
  };
  using BindingMap = std::unordered_map<uint32_t, Binding>;

  explicit ResourceBinderImpl(const Device *dev) : dev_(dev) {
  }

  std::unique_ptr<Bindings> materialize() override {
    TI_NOT_IMPLEMENTED;
    return nullptr;
  }
  // RW buffers
  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DevicePtr ptr,
                 size_t size) override {
    TI_NOT_IMPLEMENTED;
  }
  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DeviceAllocation alloc) override {
    bind_buffer(set, binding, alloc, /*is_constant=*/false);
  }

  // Constant buffers
  void buffer(uint32_t set,
              uint32_t binding,
              DevicePtr ptr,
              size_t size) override {
    TI_NOT_IMPLEMENTED;
  }
  void buffer(uint32_t set, uint32_t binding, DeviceAllocation alloc) override {
    bind_buffer(set, binding, alloc, /*is_constant=*/true);
  }

  const BindingMap &binding_map() const {
    return binding_map_;
  }

 private:
  void bind_buffer(uint32_t set,
                   uint32_t binding,
                   DeviceAllocation alloc,
                   bool is_constant) {
    TI_ASSERT(set == 0);
    TI_ASSERT(alloc.device == dev_);
    binding_map_[binding] = {alloc.alloc_id, is_constant};
  }

  const Device *const dev_;
  BindingMap binding_map_;
};

class PipelineImpl : public Pipeline {
 public:
  explicit PipelineImpl(nsobj_unique_ptr<MTLComputePipelineState> pipeline)
      : pipeline_state_(pipeline) {
  }

  ResourceBinder *resource_binder() override {
    // TODO: Hmm, why do we need this interface?
    return nullptr;
  }

  MTLComputePipelineState *mtl_pipeline_state() {
    return pipeline_state_.get();
  }

 private:
  nsobj_unique_ptr<MTLComputePipelineState> pipeline_state_{nullptr};
};

class CommandListImpl : public CommandList {
 private:
  struct ComputeEncoderBuilder {
    MTLComputePipelineState *pipeline{nullptr};
    ResourceBinderImpl::BindingMap binding_map;
  };

 public:
  explicit CommandListImpl(nsobj_unique_ptr<MTLCommandBuffer> cb)
      : command_buffer_(std::move(cb)) {
  }

  MTLCommandBuffer *command_buffer() {
    return command_buffer_.get();
  }

  void set_label(const std::string &label) {
    inflight_label_ = label;
  }

  void bind_pipeline(Pipeline *p) override {
    get_or_make_compute_builder()->pipeline =
        static_cast<PipelineImpl *>(p)->mtl_pipeline_state();
  }

  void bind_resources(ResourceBinder *binder) override {
    get_or_make_compute_builder()->binding_map =
        static_cast<ResourceBinderImpl *>(binder)->binding_map();
  }

  void bind_resources(ResourceBinder *binder,
                      ResourceBinder::Bindings *bindings) override {
    TI_NOT_IMPLEMENTED;
  }
  void buffer_barrier(DevicePtr ptr, size_t size) override {
    TI_NOT_IMPLEMENTED;
  }
  void buffer_barrier(DeviceAllocation alloc) override {
    TI_NOT_IMPLEMENTED;
  }
  void memory_barrier() override {
    TI_NOT_IMPLEMENTED;
  }

  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) override {
  }
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) override {
  }
  void dispatch(uint32_t x, uint32_t y, uint32_t z) override {
    TI_ERROR("Please call dispatch(grid_size, block_size) instead");
  }

  void dispatch(CommandList::ComputeSize grid_size,
                CommandList::ComputeSize block_size) override {
    auto encoder = new_compute_command_encoder(command_buffer_.get());
    TI_ASSERT(encoder != nullptr);
    metal::set_label(encoder.get(), inflight_label_);
    const auto &builder = inflight_compute_builder_.value();
    set_compute_pipeline_state(encoder.get(), builder.pipeline);
    auto ceil_div = [](uint32_t a, uint32_t b) -> uint32_t {
      return (a + b - 1) / b;
    };
    const auto num_blocks_x = ceil_div(grid_size.x, block_size.x);
    const auto num_blocks_y = ceil_div(grid_size.y, block_size.y);
    const auto num_blocks_z = ceil_div(grid_size.z, block_size.z);
    dispatch_threadgroups(encoder.get(), num_blocks_x, num_blocks_y,
                          num_blocks_z, block_size.x, block_size.y,
                          block_size.z);
    finish_encoder(encoder.get());
  }

  // Graphics commands are not implemented on Metal
 private:
  ComputeEncoderBuilder *get_or_make_compute_builder() {
    if (!inflight_compute_builder_.has_value()) {
      inflight_compute_builder_ = ComputeEncoderBuilder{};
    }
    return &(inflight_compute_builder_.value());
  }

  template <typename T>
  void finish_encoder(T *encoder) {
    end_encoding(encoder);
    inflight_label_.clear();
    inflight_compute_builder_.reset();
  }

  nsobj_unique_ptr<MTLCommandBuffer> command_buffer_{nullptr};
  std::string inflight_label_;
  std::optional<ComputeEncoderBuilder> inflight_compute_builder_;
};

class StreamImpl : public Stream {
 public:
  explicit StreamImpl(MTLCommandQueue *command_queue)
      : command_queue_(command_queue) {
  }

  std::unique_ptr<CommandList> new_command_list() override {
    auto cb = new_command_buffer(command_queue_);
    TI_ASSERT(cb != nullptr);
    set_label(cb.get(), fmt::format("command_buffer_{}", list_counter_++));
    return std::make_unique<CommandListImpl>(std::move(cb));
  }

  void submit(CommandList *cmdlist) override {
    auto *cb = static_cast<CommandListImpl *>(cmdlist)->command_buffer();
    commit_command_buffer(cb);
  }
  void submit_synced(CommandList *cmdlist) override {
    auto *cb = static_cast<CommandListImpl *>(cmdlist)->command_buffer();
    commit_command_buffer(cb);
    wait_until_completed(cb);
  }

  void command_sync() override {
    // No-op on Metal
  }

 private:
  MTLCommandQueue *const command_queue_;
  uint32_t list_counter_{0};
};

class DeviceImpl : public Device {
 public:
  explicit DeviceImpl(MTLDevice *dev) : device_(dev) {
    command_queue_ = new_command_queue(device_);
    TI_ASSERT(command_queue_ != nullptr);
    // TODO: thread local streams?
    stream_ = std::make_unique<StreamImpl>(command_queue_.get());
    TI_ASSERT(stream_ != nullptr);
  }

  DeviceAllocation allocate_memory(const AllocParams &params) override {
    TI_NOT_IMPLEMENTED;
  }

  void dealloc_memory(DeviceAllocation handle) override {
    TI_NOT_IMPLEMENTED;
  }

  std::unique_ptr<Pipeline> create_pipeline(const PipelineSourceDesc &src,
                                            std::string name) override {
    TI_ASSERT(src.type == PipelineSourceType::metal_src);
    TI_ASSERT(src.stage == PipelineStageType::compute);
    // FIXME: infer version/fast_math
    std::string src_code{static_cast<char *>(src.data), src.size};
    auto kernel_lib = new_library_with_source(
        device_, src_code, /*fast_math=*/false, kMslVersionNone);
    TI_ASSERT(kernel_lib != nullptr);
    auto mtl_func = new_function_with_name(kernel_lib.get(), name);
    TI_ASSERT(mtl_func != nullptr);
    auto pipeline =
        new_compute_pipeline_state_with_function(device_, mtl_func.get());
    TI_ASSERT(pipeline != nullptr);
    return std::make_unique<PipelineImpl>(std::move(pipeline));
  }

  void *map_range(DevicePtr ptr, uint64_t size) override {
    TI_NOT_IMPLEMENTED;
  }
  void *map(DeviceAllocation alloc) override {
    TI_NOT_IMPLEMENTED;
  }

  void unmap(DevicePtr ptr) override {
    TI_NOT_IMPLEMENTED;
  }
  void unmap(DeviceAllocation alloc) override {
    TI_NOT_IMPLEMENTED;
  }

  Stream *get_compute_stream() override {
    return stream_.get();
  }

 private:
  MTLDevice *const device_;
  nsobj_unique_ptr<MTLCommandQueue> command_queue_{nullptr};
  std::unique_ptr<StreamImpl> stream_{nullptr};
};

}  // namespace

std::unique_ptr<taichi::lang::Device> make_compute_device() {
  return std::make_unique<DeviceImpl>();
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
