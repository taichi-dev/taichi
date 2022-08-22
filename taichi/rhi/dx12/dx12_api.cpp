#include "taichi/rhi/dx12/dx12_api.h"

namespace taichi {
namespace lang {
namespace directx12 {

bool is_dx12_api_available() {
#ifdef TI_WITH_DX12
  return true;
#else
  return false;
#endif
}

std::shared_ptr<Device> make_dx12_device() {
  return nullptr;
}

std::vector<uint8_t> validate_and_sign(
    std::vector<uint8_t> &input_dxil_container) {
  // FIXME: implement validation and sign by calling IDxcValidator::Validate
  // from DXIL.dll.
  return input_dxil_container;
}

}  // namespace directx12
}  // namespace lang
}  // namespace taichi
