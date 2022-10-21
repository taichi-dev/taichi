#pragma once

#include <vector>
#include <map>

#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/aot/module_loader.h"

namespace taichi::lang {
namespace gfx {

/**
 * AOT module data for the Unified Device API backend.
 */
struct TaichiAotData {
  //   BufferMetaData metadata;
  std::vector<std::vector<std::vector<uint32_t>>> spirv_codes;
  std::vector<spirv::TaichiKernelAttributes> kernels;
  std::vector<aot::CompiledFieldData> fields;
  std::map<std::string, uint32_t> required_caps;
  size_t root_buffer_size{0};

  TI_IO_DEF(kernels, fields, required_caps, root_buffer_size);
};

}  // namespace gfx
}  // namespace taichi::lang
