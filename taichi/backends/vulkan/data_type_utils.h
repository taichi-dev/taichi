#pragma once

#include <algorithm>
#include <cstddef>

#include "taichi/lang_util.h"

namespace taichi {
namespace lang {
namespace vulkan {

inline std::size_t vk_data_type_size(DataType dt) {
  return data_type_size(dt);
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
