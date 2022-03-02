#pragma once

#include <memory>

#include "taichi/backends/device.h"

namespace taichi {
namespace lang {
namespace metal {

std::unique_ptr<taichi::lang::Device> make_compute_device();

}  // namespace metal
}  // namespace lang
}  // namespace taichi
