#pragma once
#include <set>
#include "taichi/platform/mac/objc_api.h"
#include "taichi/rhi/device.h"
#include "taichi/rhi/metal/metal_api.h"

namespace taichi::lang::metal {

struct MetalStream : public Stream {
 public:
  explicit MetalStream(
    MetalDevice* device,
    mac::nsobj_unique_ptr<MTL::CommandQueue>&& command_queue
  );

  static std::unique_ptr<MetalStream> create(MetalDevice* device);

  std::unique_ptr<CommandList> new_command_list() override;

  StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores) override;
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores) override;
  void command_sync() override;

  constexpr MetalDevice* get_device() const {
    return device_;
  }
  constexpr MTL::CommandQueue *get_mtl_command_queue() const {
    return command_queue_.get();
  }

 private:
  MetalDevice *device_;
  mac::nsobj_unique_ptr<MTL::CommandQueue> command_queue_;

  // (penguinliong) Waited on and cleared during synchronization.
  std::vector<mac::nsobj_unique_ptr<MTL::CommandBuffer>> pending_command_buffers_;
};

} // namespace taichi::lang::metal
