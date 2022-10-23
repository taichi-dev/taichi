#include "taichi/platform/mac/objc_api.h"
#include "taichi/rhi/metal/metal_device.h"
#include "taichi/rhi/metal/metal_stream.h"
#include "taichi/rhi/metal/metal_command_list.h"

namespace taichi::lang::metal {

MetalStream::MetalStream(
  MetalDevice* device,
  mac::nsobj_unique_ptr<MTL::CommandQueue>&& command_queue
) : device_(device), command_queue_(std::move(command_queue)) {
}

std::unique_ptr<MetalStream> MetalStream::create(MetalDevice* device) {
  TI_ASSERT(device != nullptr);
  mac::nsobj_unique_ptr<MTL::CommandQueue> command_queue =
    mac::wrap_as_nsobj_unique_ptr(device->get_mtl_device()->newCommandQueue());
  return std::make_unique<MetalStream>(device, std::move(command_queue));
}

std::unique_ptr<CommandList> MetalStream::new_command_list() {
  return std::unique_ptr<CommandList>(MetalCommandList::create(this).release());
}

StreamSemaphore MetalStream::submit(
    CommandList *cmdlist,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  MetalCommandList *cmdlist2 = static_cast<MetalCommandList *>(cmdlist);
  cmdlist2->get_mtl_command_buffer()->commit();
  pending_command_buffers_.emplace_back(
    mac::retain_and_wrap_as_nsobj_unique_ptr(cmdlist2->get_mtl_command_buffer()));

  // FIXME: Implement semaphore mechanism for Metal backend
  //        and return the actual semaphore corresponding to the submitted
  //        cmds.
  return nullptr;
}
StreamSemaphore MetalStream::submit_synced(
    CommandList *cmdlist,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  submit(cmdlist, wait_semaphores);
  command_sync();

  // (penguinliong) There is no semaphore in Metal. Can be impled with events
  // tho.
  return nullptr;
}

void MetalStream::command_sync() {
  for (const auto& cmdlist : pending_command_buffers_) {
    cmdlist->waitUntilCompleted();
  }
  pending_command_buffers_.clear();
}

} // namespace taichi::lang::metal
