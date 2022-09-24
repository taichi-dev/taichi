#pragma once

#include <vector>

#include "taichi/runtime/metal/kernel_utils.h"

namespace taichi::lang {
namespace metal {

/**
 * AOT module data for the metal backend.
 */
struct TaichiAotData {
  BufferMetaData metadata;
  std::vector<CompiledKernelData> kernels;
  std::vector<CompiledKernelTmplData> tmpl_kernels;
  std::vector<CompiledFieldData> fields;

  TI_IO_DEF(metadata, kernels, tmpl_kernels, fields);
};

}  // namespace metal
}  // namespace taichi::lang
