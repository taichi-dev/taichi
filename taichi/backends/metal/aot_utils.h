#pragma once

#include <vector>

#include "taichi/backends/metal/kernel_utils.h"

namespace taichi {
namespace lang {
namespace metal {

/**
 * AOT module data for the metal backend.
 */
struct TaichiAotData {
  BufferMetaData metadata;
  std::vector<CompiledKernelData> kernels;

  TI_IO_DEF(metadata, kernels);
};

}  // namespace metal
}  // namespace lang
}  // namespace taichi
