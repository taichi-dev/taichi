#include "taichi/ui/backends/vulkan/vertex.h"

#include <stddef.h>

namespace taichi {
namespace ui {

size_t sizeof_vbo(VboAttribes va) {
  switch (va) {
    case VboAttribes::kAll:
      return sizeof(Vertex);
    case VboAttribes::kPos:
      return offsetof(Vertex, normal);
    case VboAttribes::kPosNormal:
      return offsetof(Vertex, texCoord);
    case VboAttribes::kPosNormalUv:
      return offsetof(Vertex, color);
    default:
      break;
  }
}

}  // namespace ui
}  // namespace taichi
