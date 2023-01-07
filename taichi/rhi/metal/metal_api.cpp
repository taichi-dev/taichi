#include "taichi/rhi/metal/metal_api.h"
#include "taichi/rhi/metal/metal_device.h"

namespace taichi::lang {
namespace metal {

bool is_metal_api_available() {
#if defined(__APPLE__) && defined(TI_WITH_METAL)
  return true;
#else
  return false;
#endif  // defined(__APPLE__) && defined(TI_WITH_METAL)
}

std::shared_ptr<Device> create_metal_device() {
#if defined(__APPLE__) && defined(TI_WITH_METAL)
  return std::shared_ptr<Device>(metal::MetalDevice::create());
#else
  return nullptr;
#endif  // defined(__APPLE__) && defined(TI_WITH_METAL)
}

}  // namespace metal
}  // namespace taichi::lang
