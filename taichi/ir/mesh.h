#pragma once

#include <atomic>

#include "taichi/ir/type.h"
#include "taichi/ir/snode.h"

namespace taichi {
namespace lang {
namespace mesh {

enum class MeshElementType {
  Vertex = 0,
  Edge = 1,
  Face = 2,
  Cell = 3
};

enum class MeshRelationType {
  VV = 0, VE = 1, VF = 2, VC = 3,
  EV = 4, EE = 5, EF = 6, EC = 7,
  FV = 8, FE = 9, FF = 10, FC = 11,
  CV = 12, CE = 13, CF = 14, CC = 15,
};

struct MeshAttribute {
 public:
  MeshAttribute() {}
};

struct MeshLocalRelation {
 public:
  MeshLocalRelation() {}
};

class Mesh {
 public:
  Mesh() {
    owned_offset.clear();
    total_offset.clear();
  }

  uint32_t num_patches {0};

  template<typename T>
  using MeshMapping = std::unordered_map<MeshElementType, T>;

  MeshMapping<SNode*> owned_offset{}; // prefix of owned element
  MeshMapping<SNode*> total_offset{}; // prefix of total element
  MeshMapping<SNode*> l2g_map{}; // local to global index mapping

  MeshMapping<std::size_t /*offset*/> owned_offset_local;
  MeshMapping<std::size_t /*offset*/> total_offset_local;
  MeshMapping<std::size_t /*offset*/> owned_num_local;
  MeshMapping<std::size_t /*offset*/> total_num_local;

  std::map<MeshRelationType, MeshLocalRelation> relations;

  std::unordered_map</*snode_id*/ int, MeshAttribute> attribute;
};

struct MeshPtr { // Mesh wrapper in python
  std::shared_ptr<Mesh> ptr;
};

} /*mesh*/
} /*lang*/
} /*taichi*/