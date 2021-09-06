#include "taichi/ir/mesh.h"

namespace taichi {
namespace lang {
namespace mesh {

const char *element_type_str(MeshElementType type) {
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

const char *conv_type_str(ConvType type) {
  if (type == mesh::ConvType::l2g)
    return "local to global";
  else if (type == mesh::ConvType::l2r)
    return "local to reordered";
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

MeshRelationType relation_by_orders(int from_order, int to_order) {
  return MeshRelationType((from_order << 2) | to_order);
}

MeshRelationType inverse_relation(MeshRelationType rel) {
  return relation_by_orders(to_end_element_order(rel),
                            from_end_element_order(rel));
}

}  // namespace mesh
}  // namespace lang
}  // namespace taichi
