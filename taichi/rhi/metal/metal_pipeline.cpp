#include "taichi/rhi/metal/metal_pipeline.h"
#include "taichi/rhi/metal/metal_device.h"
#include "taichi/rhi/metal/metal_resource_binder.h"

namespace taichi::lang::metal {

using mac::nsobj_unique_ptr;

MetalPipeline::MetalPipeline(
  MetalDevice* device,
  mac::nsobj_unique_ptr<MTL::Library> library,
  mac::nsobj_unique_ptr<MTL::Function> function,
  mac::nsobj_unique_ptr<MTL::ComputePipelineState>&& compute_pipeline_state
) : device_(device), library_(std::move(library)), function_(std::move(function)), compute_pipeline_state_(std::move(compute_pipeline_state)) {
}

std::unique_ptr<MetalPipeline> MetalPipeline::create(MetalDevice* device, const PipelineSourceDesc& src, const std::string& name) {
  TI_ASSERT(src.type == PipelineSourceType::metal_src);
  TI_ASSERT(src.stage == PipelineStageType::compute);
  // FIXME: infer version/fast_math
  std::string src2(static_cast<const char *>(src.data), src.size);
  nsobj_unique_ptr<NS::String> src3 =
      mac::wrap_string_as_ns_string(src2.c_str());

  NS::Error* err = nullptr;

  nsobj_unique_ptr<MTL::CompileOptions> compile_opts =
      mac::wrap_as_nsobj_unique_ptr(MTL::CompileOptions::alloc()->init());
  compile_opts->setFastMathEnabled(device->get_cap(DeviceCapability::metal_fast_math) != 0);
  // This assertion is too aggressive but is to help me remember there is a
  // switch for MSL language version.
  TI_ASSERT(device->get_cap(DeviceCapability::metal_msl_version) == (uint32_t)MTL::LanguageVersion2_4);
  compile_opts->setLanguageVersion(
      (MTL::LanguageVersion)device->get_cap(DeviceCapability::metal_msl_version));

  nsobj_unique_ptr<MTL::Library> library = mac::wrap_as_nsobj_unique_ptr(
      device->get_mtl_device()->newLibrary(src3.get(), compile_opts.get(), &err));
  if (library == nullptr) {
    TI_ERROR("cannot create metal kernel library: {}",
        mac::to_string(err->description()));
    err->release();
    err = nullptr;
    return nullptr;
  }

  nsobj_unique_ptr<MTL::Function> function = mac::wrap_as_nsobj_unique_ptr(
      library->newFunction(mac::wrap_string_as_ns_string(name).get()));
  if (function == nullptr) {
    TI_ERROR("cannot create metal kernel function");
    return nullptr;
  }

  nsobj_unique_ptr<MTL::ComputePipelineState> pipe_state = mac::wrap_as_nsobj_unique_ptr(
      device->get_mtl_device()->newComputePipelineState(function.get(), &err));
  if (pipe_state == nullptr) {
    TI_ERROR("cannot create metal pipeline: {}",
        mac::to_string(err->description()));
    err->release();
    err = nullptr;
    return nullptr;
  }

  return std::make_unique<MetalPipeline>(
      device, std::move(library), std::move(function), std::move(pipe_state));
}

ResourceBinder *MetalPipeline::resource_binder() {
  return binder_.get();
}

} // namespace taichi::lang::metal
