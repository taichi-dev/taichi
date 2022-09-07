#pragma once
#include "taichi_gfx_impl.h"
#include "taichi/taichi_opengl.h"
#include "taichi/rhi/opengl/opengl_device.h"

class OpenglRuntime : public GfxRuntime {
 private:
  taichi::lang::opengl::GLDevice device_;
  taichi::lang::gfx::GfxRuntime gfx_runtime_;

 public:
  OpenglRuntime();
  virtual taichi::lang::Device &get() override final;
  virtual taichi::lang::gfx::GfxRuntime &get_gfx_runtime() override final;
  taichi::lang::opengl::GLDevice &get_gl() {
    return device_;
  }
};
