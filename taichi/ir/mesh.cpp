#include "taichi/ir/mesh.h"

namespace taichi {
namespace lang {
namespace mesh {

std::string element_type_name(MeshElementType type) {
  if (type == MeshElementType::Vertex)
    return "verts";
  else if (type == MeshElementType::Edge)
    return "edges";
  else if (type == MeshElementType::Face)
    return "faces";
  else if (type == MeshElementType::Cell)
    return "cells";
  else {
    TI_NOT_IMPLEMENTED;
  }
}

std::string relation_type_name(MeshRelationType type) {
  return element_type_name(MeshElementType(from_end_element_order(type))) +
         "-" + element_type_name(MeshElementType(to_end_element_order(type)));
}

std::string conv_type_name(ConvType type) {
  if (type == mesh::ConvType::l2g)
    return "local to global";
  else if (type == mesh::ConvType::l2r)
    return "local to reordered";
  else if (type == mesh::ConvType::g2r) {
    return "global to reordered";
  } else {
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
  return int(rel) & 0x3;
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
