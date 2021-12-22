#include "taichi/backends/dx/dx_device.h"

namespace taichi {
namespace lang {
namespace directx11 {

Dx11ResourceBinder::~Dx11ResourceBinder() {
}

Dx11Pipeline::Dx11Pipeline(const PipelineSourceDesc &desc,
                           const std::string &name) {
  TI_NOT_IMPLEMENTED;
}

Dx11Pipeline::~Dx11Pipeline() {
}

ResourceBinder *Dx11Pipeline::resource_binder() {
  return nullptr;
}

}
}  // namespace lang
}  // namespace taichi