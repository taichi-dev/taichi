// Codegen for the hierarchical data structure
#pragma once

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "taichi/ir/snode.h"
#include "taichi/platform/metal/metal_data_types.h"
#include "taichi/platform/metal/metal_kernel_util.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

struct StructCompiledResult {
  // Source code of the SNode data structures compiled to Metal
  std::string source_code;
  // Root buffer size in bytes.
  size_t root_size;
};

// Compile all snodes to Metal source code
StructCompiledResult compile_structs(SNode &root);

}  // namespace metal
TLANG_NAMESPACE_END
