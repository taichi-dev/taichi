#include "taichi/ir/mesh.h"

namespace taichi {
namespace lang {
namespace mesh {

const char * element_type_str(MeshElementType type) {
  if (type == MeshElementType::Vertex)
    return "Vertex";
  else if (type == MeshElementType::Edge)
    return "Edge";
  else if (type == MeshElementType::Face)
    return "Face";
  else if (type == MeshElementType::Cell)
    return "Cell";
  else {
    TI_NOT_IMPLEMENTED;
  }
}

int element_order(MeshElementType type) {
  return int(type);
}

int from_end_element_order(MeshRelationType rel) {
  return int(rel) >> 0x2;
}

int to_end_element_order(MeshRelationType rel) {
  return int(rel) & 0x4;
}

int relation_by_orders(int from_order, int to_order) {
  return ((from_order << 2) | to_order);
}

int inverse_relation(MeshRelationType rel) {
  return relation_by_orders(to_end_element_order(rel), from_end_element_order(rel));
}

} /*mesh*/
} /*lang*/
} /*taichi*/