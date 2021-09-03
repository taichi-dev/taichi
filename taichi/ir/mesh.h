#pragma once

#include <atomic>

#include "taichi/ir/type.h"
#include "taichi/ir/snode.h"

#include <unordered_set>

namespace taichi {
namespace lang {

class Stmt;

namespace mesh {

enum class MeshTopology { Triangle = 3, Tetrahedron = 4 };

enum class MeshElementType { Vertex = 0, Edge = 1, Face = 2, Cell = 3 };

const char *element_type_str(MeshElementType type);

enum class MeshRelationType {
  VV = 0,
  VE = 1,
  VF = 2,
  VC = 3,
  EV = 4,
  EE = 5,
  EF = 6,
  EC = 7,
  FV = 8,
  FE = 9,
  FF = 10,
  FC = 11,
  CV = 12,
  CE = 13,
  CF = 14,
  CC = 15,
};

enum class MeshElementReorderingType {
  NonReordering = 0,
  Reordering = 1,
  SurfaceFirst = 2,
  CellFirst = 3
};

enum class ConvType { l2g, l2r };

const char *conv_type_str(ConvType type);

int element_order(MeshElementType type);
int from_end_element_order(MeshRelationType rel);
int to_end_element_order(MeshRelationType rel);
int relation_by_orders(int from_order, int to_order);
int inverse_relation(MeshRelationType rel);

struct MeshAttribute {
  MeshAttribute(MeshElementType type_,
                SNode *snode_,
                MeshElementReorderingType reordering_type_)
      : type(type_), snode(snode_), reordering_type(reordering_type_) {
  }

  bool operator==(const MeshAttribute &rhs) const {
    return type == rhs.type && snode == rhs.snode &&
           reordering_type == rhs.reordering_type;
  }

  struct Hash {
    size_t operator()(const MeshAttribute &attr) const {
      uintptr_t ad = (uintptr_t)attr.snode;
      return (size_t)((13 * ad) ^ (ad >> 15));
    }
  };

  MeshElementType type;
  SNode *snode;
  MeshElementReorderingType reordering_type;
};

struct MeshLocalRelation {
  MeshLocalRelation() {
  }
};

class Mesh {
 public:
  Mesh() {
  }

  uint32_t num_patches{0};

  template <typename T>
  using MeshMapping = std::unordered_map<MeshElementType, T>;

  MeshMapping<SNode *> owned_offset{};  // prefix of owned element
  MeshMapping<SNode *> total_offset{};  // prefix of total element
  MeshMapping<SNode *> l2g_map{};       // local to global index mapping

  std::unordered_map<mesh::MeshElementType, Stmt *>
      owned_offset_local;  // |owned_offset[idx]|
  std::unordered_map<mesh::MeshElementType, Stmt *>
      total_offset_local;  // |total_offset[idx]|
  std::unordered_map<mesh::MeshElementType, Stmt *>
      owned_num_local;  // |owned_offset[idx+1] - owned_offset[idx]|
  std::unordered_map<mesh::MeshElementType, Stmt *>
      total_num_local;  // |total_offset[idx+1] - total_offset[idx]|

  MeshMapping<std::unordered_set<MeshAttribute, MeshAttribute::Hash>>
      attributes;
  std::map<MeshRelationType, MeshLocalRelation> relations;
};

struct MeshPtr {  // Mesh wrapper in python
  std::shared_ptr<Mesh> ptr;
};

}  // namespace mesh
}  // namespace lang
}  // namespace taichi
