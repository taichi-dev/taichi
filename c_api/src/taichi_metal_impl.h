#pragma once
#ifdef TI_WITH_METAL
#include "taichi_core_impl.h"
#include "taichi_gfx_impl.h"
#include "taichi/rhi/metal/metal_device.h"

namespace capi {

class MetalRuntime;

class MetalRuntime : public GfxRuntime {
 private:
  std::unique_ptr<taichi::lang::metal::MetalDevice> mtl_device_;
  taichi::lang::gfx::GfxRuntime gfx_runtime_;

 public:
  explicit MetalRuntime();
  explicit MetalRuntime(
      std::unique_ptr<taichi::lang::metal::MetalDevice> &&mtl_device);

  taichi::lang::Device &get() override;
  taichi::lang::gfx::GfxRuntime &get_gfx_runtime() override;

  taichi::lang::metal::MetalDevice &get_mtl();
  virtual TiImage allocate_image(
      const taichi::lang::ImageParams &params) override final;
  virtual void free_image(TiImage image) override final;
};

}  // namespace capi

#endif  // TI_WITH_METAL
