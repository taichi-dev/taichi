#pragma once

#include "taichi/backends/device.h"

namespace taichi {
namespace lang {
namespace directx11 {

class Dx11ResourceBinder : public ResourceBinder {
  ~Dx11ResourceBinder() override;
};

class Dx11Pipeline : public Pipeline {
 public:
  Dx11Pipeline(const PipelineSourceDesc &desc, const std::string &name);
  ~Dx11Pipeline() override;
  ResourceBinder *resource_binder() override;
};

}  // namespace directx11
}  // namespace lang
}  // namespace taichi