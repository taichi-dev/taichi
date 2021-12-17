#pragma once

#include <vector>

#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/program/aot_module.h"

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
  std::vector<aot::CompiledFieldData> fields;
  size_t root_buffer_size;

  TI_IO_DEF(kernels, fields, root_buffer_size);
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
