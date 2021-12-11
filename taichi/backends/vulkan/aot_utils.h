#pragma once

#include <vector>

#include "taichi/codegen/spirv/kernel_utils.h"

namespace taichi {
namespace lang {
namespace vulkan {

/**
 * AOT module data for the vulkan backend.
 */
struct TaichiAotData {
  //   BufferMetaData metadata;
  std::vector<std::vector<std::vector<uint32_t>>> spirv_codes;
  std::vector<spirv::TaichiKernelAttributes> kernels;
  std::vector<spirv::CompiledFieldData> fields;

  TI_IO_DEF(kernels, fields);
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
