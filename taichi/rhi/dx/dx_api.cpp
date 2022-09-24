#include "taichi/rhi/dx/dx_api.h"
#include "taichi/rhi/dx/dx_device.h"

namespace taichi::lang {
namespace directx11 {

bool is_dx_api_available() {
#ifdef TI_WITH_DX11
  return true;
#else
  return false;
#endif
}

std::shared_ptr<Device> make_dx11_device() {
  return std::make_shared<directx11::Dx11Device>();
}

}  // namespace directx11
}  // namespace taichi::lang
