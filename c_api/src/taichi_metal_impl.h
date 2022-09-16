#pragma once
#include "taichi/rhi/metal/device.h"
#include "taichi/codegen/metal/struct_metal.h"
#include "taichi/runtime/metal/kernel_manager.h"

#include "taichi_core_impl.h"
#include "taichi_gfx_impl.h"

class MetalRuntime;

class MetalRuntime : public GfxRuntime {
 public:
  MetalRuntime();
};
class MetalRuntimeOwned : public MetalRuntime {
  std::unique_ptr<taichi::lang::metal::CompiledRuntimeModule> runtime_module_;
  std::unique_ptr<taichi::lang::metal::KernelManager> kernel_manager_;
  std::unique_ptr<taichi::lang::Device> device_;
  taichi::lang::gfx::GfxRuntime gfx_runtime_;

 public:
  MetalRuntimeOwned();

  virtual taichi::lang::Device &get() override final;
  virtual taichi::lang::gfx::GfxRuntime &get_gfx_runtime() override final;
};
