#pragma once

#include <atomic>

#include "taichi/ir/type.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/scratch_pad.h"

#include <unordered_set>

namespace taichi {
namespace lang {

class Stmt;

namespace mesh {

enum class MeshTopology { Triangle = 3, Tetrahedron = 4 };

enum class MeshElementType { Vertex = 0, Edge = 1, Face = 2, Cell = 3 };

std::string element_type_name(MeshElementType type);

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

std::string relation_type_name(MeshRelationType type);

enum class ConvType { l2g, l2r, g2r };

std::string conv_type_name(ConvType type);

int element_order(MeshElementType type);
int from_end_element_order(MeshRelationType rel);
int to_end_element_order(MeshRelationType rel);
MeshRelationType relation_by_orders(int from_order, int to_order);
MeshRelationType inverse_relation(MeshRelationType rel);

struct MeshLocalRelation {
  MeshLocalRelation(SNode *value_, SNode *patch_offset_, SNode *offset_, int max_value_per_patch_)
      : value(value_), patch_offset(patch_offset_), offset(offset_), max_value_per_patch(max_value_per_patch_) {
    fixed = false;
  }

  MeshLocalRelation(SNode *value_, int max_value_per_patch_) : value(value_), max_value_per_patch(max_value_per_patch_) {
    fixed = true;
  }

  bool fixed;
  int max_value_per_patch{0};
  SNode *value{nullptr};
  SNode *patch_offset{nullptr};
  SNode *offset{nullptr};
};

class Mesh {
 public:
  Mesh() = default;

  template <typename T>
  using MeshMapping = std::unordered_map<MeshElementType, T>;

  int num_patches{0};
  MeshMapping<int> num_elements{};
  MeshMapping<int>
      patch_max_element_num{};  // the max number of mesh element in each patch

  MeshMapping<SNode *> owned_offset{};  // prefix of owned element
  MeshMapping<SNode *> total_offset{};  // prefix of total element
  std::map<std::pair<MeshElementType, ConvType>, SNode *>
      index_mapping{};  // mapping from one index space to another index space

  std::map<MeshRelationType, MeshLocalRelation> relations;
};

struct MeshPtr {  // Mesh wrapper in python
  std::shared_ptr<Mesh> ptr;
};

}  // namespace mesh
}  // namespace lang
}  // namespace taichi
