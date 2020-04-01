#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include <taichi/ir/statements.h>

TLANG_NAMESPACE_BEGIN

class SNode;

namespace opengl {

struct StructCompiledResult {
  // Source code of the SNode data structures compiled to GLSL
  std::string source_code;
  std::unordered_map<std::string, std::vector<size_t>> class_get_map;
  std::unordered_map<std::string, size_t> class_children_map;
  // Root buffer size in bytes.
  size_t root_size;
};

struct IOV {
  void *base;
  size_t size;
};

}  // namespace opengl

TLANG_NAMESPACE_END
