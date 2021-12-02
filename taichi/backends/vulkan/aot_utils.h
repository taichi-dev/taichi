#pragma once

#include <vector>

#include "taichi/backends/vulkan/kernel_utils.h"

namespace taichi {
namespace lang {
namespace vulkan {

/**
 * AOT module data for the vulkan backend.
 */
struct TaichiAotData {
  //   BufferMetaData metadata;
  std::vector<std::vector<std::vector<uint32_t>>> spirv_codes;
  std::vector<TaichiKernelAttributes> kernels;

  TI_IO_DEF(kernels);
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
