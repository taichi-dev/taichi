#include "taichi/ui/backends/vulkan/vertex.h"

#include <stddef.h>

namespace taichi {
namespace ui {

// static
size_t VboHelpers::size(VertexAttributes va) {
  size_t res = 0;
  if (VboHelpers::has_attr(va, VertexAttributes::kPos)) {
    res += sizeof(Vertex::pos);
  }
  if (VboHelpers::has_attr(va, VertexAttributes::kNormal)) {
    res += sizeof(Vertex::normal);
  }
  if (VboHelpers::has_attr(va, VertexAttributes::kUv)) {
    res += sizeof(Vertex::tex_coord);
  }
  if (VboHelpers::has_attr(va, VertexAttributes::kColor)) {
    res += sizeof(Vertex::color);
  }
  return res;
}

}  // namespace ui
}  // namespace taichi
