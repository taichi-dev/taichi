#pragma once

#include <optional>
#include <string>
#include <vector>

#include "taichi/backends/metal/kernel_util.h"

TLANG_NAMESPACE_BEGIN

class Kernel;
class SNode;

namespace metal {
struct BufferSize {
  int64_t root_buffer_size;
  int64_t runtime_buffer_size;
  int64_t randseedoffset_in_runtime_buffer;

  TI_IO_DEF(root_buffer_size,
            runtime_buffer_size,
            randseedoffset_in_runtime_buffer);
};

struct MetalTiFileData {
  BufferSize sizes;
  std::vector<CompiledKernelData> kernels_;

  TI_IO_DEF(sizes, kernels_);
};

}  // namespace metal

TLANG_NAMESPACE_END
