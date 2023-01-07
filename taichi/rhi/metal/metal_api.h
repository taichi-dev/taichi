#pragma once
#include "taichi/rhi/device.h"

namespace taichi::lang {
namespace metal {

bool is_metal_api_available();

std::shared_ptr<Device> create_metal_device();

}  // namespace metal
}  // namespace taichi::lang
