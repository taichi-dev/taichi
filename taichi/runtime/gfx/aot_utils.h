#pragma once

#include <vector>

#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/aot/module_loader.h"

namespace taichi {
namespace lang {
namespace gfx {

/**
 * AOT module data for the Unified Device API backend.
 */
struct TaichiAotData {
  //   BufferMetaData metadata;
  std::vector<std::vector<std::vector<uint32_t>>> spirv_codes;
  std::vector<spirv::TaichiKernelAttributes> kernels;
  std::vector<aot::CompiledFieldData> fields;
  size_t root_buffer_size{0};

  TI_IO_DEF(kernels, fields, root_buffer_size);
};

}  // namespace gfx
}  // namespace lang
}  // namespace taichi
