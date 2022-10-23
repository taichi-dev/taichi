#pragma once
#include "taichi/platform/mac/objc_api.h"
#include "taichi/rhi/device.h"
#include "taichi/rhi/metal/metal_api.h"
// FIXME: (penguinliong) Included to workaround compilation error. But from the
// very beginning the resource binder SHOULD NOT be a member of this.
#include "taichi/rhi/metal/metal_resource_binder.h"

namespace taichi::lang::metal {

struct MetalPipeline : public Pipeline {
 public:
  explicit MetalPipeline(
    MetalDevice* device,
    mac::nsobj_unique_ptr<MTL::Library> library,
    mac::nsobj_unique_ptr<MTL::Function> function,
    mac::nsobj_unique_ptr<MTL::ComputePipelineState>&& compute_pipeline_state
  );

  static std::unique_ptr<MetalPipeline> create(MetalDevice* device, const PipelineSourceDesc& src, const std::string& name);

  ResourceBinder *resource_binder() override;

  constexpr MTL::ComputePipelineState *get_mtl_pipeline_state() const {
    return compute_pipeline_state_.get();
  }

 private:
  MetalDevice *device_;
  mac::nsobj_unique_ptr<MTL::Library> library_;
  mac::nsobj_unique_ptr<MTL::Function> function_;
  mac::nsobj_unique_ptr<MTL::ComputePipelineState> compute_pipeline_state_;

  // (penguinliong) The design of this interface is somehow problematic, i.e.,
  // we have to keep this binder as a field and it doesn't allow us to reuse a
  // pipeline among multiple threads.
  //
  // Well, I know we are unlikely to support multithreading recently but it
  // simply doesn't sound good.
  std::unique_ptr<MetalResourceBinder> binder_;
};

} // namespace taichi::lang::metal
