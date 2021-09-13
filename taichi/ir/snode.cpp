#include "taichi/ir/snode.h"

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

std::atomic<int> SNode::counter{0};

SNode &SNode::insert_children(SNodeType t) {
  TI_ASSERT(t != SNodeType::root);

  auto new_ch = std::make_unique<SNode>(depth + 1, t);
  new_ch->parent = this;
  new_ch->is_path_all_dense = (is_path_all_dense && !new_ch->need_activation());
  for (int i = 0; i < taichi_max_num_indices; i++) {
    new_ch->extractors[i].num_elements_from_root *=
        extractors[i].num_elements_from_root;
  }
  std::memcpy(new_ch->physical_index_position, physical_index_position,
              sizeof(physical_index_position));
  new_ch->num_active_indices = num_active_indices;
  if (type == SNodeType::bit_struct || type == SNodeType::bit_array) {
    new_ch->is_bit_level = true;
  } else {
    new_ch->is_bit_level = is_bit_level;
  }
  ch.push_back(std::move(new_ch));
  return *ch.back();
}

SNode &SNode::create_node(std::vector<Axis> axes,
                          std::vector<int> sizes,
                          SNodeType type,
                          bool packed) {
  TI_ASSERT(axes.size() == sizes.size() || sizes.size() == 1);
  if (sizes.size() == 1) {
    sizes = std::vector<int>(axes.size(), sizes[0]);
  }

  if (type == SNodeType::hash)
    TI_ASSERT_INFO(depth == 0,
                   "hashed node must be child of root due to initialization "
                   "memset limitation.");

  auto &new_node = insert_children(type);
  for (int i = 0; i < (int)axes.size(); i++) {
    TI_ASSERT(sizes[i] > 0);
    auto &ind = axes[i];
    new_node.extractors[ind.value].activate(
        bit::log2int(bit::least_pot_bound(sizes[i])));
    new_node.extractors[ind.value].num_elements_from_root *= sizes[i];
    if (packed) {
      new_node.extractors[ind.value].shape = sizes[i];
    } else {  // if not in packed mode, pad shape to POT
      new_node.extractors[ind.value].shape =
          1 << new_node.extractors[ind.value].num_bits;
    }
  }
  // infer mappings
  for (int i = 0; i < taichi_max_num_indices; i++) {
    bool found = false;
    for (int k = 0; k < taichi_max_num_indices; k++) {
      if (new_node.physical_index_position[k] == i) {
        found = true;
        break;
      }
    }
    if (found)
      continue;
    if (new_node.extractors[i].active) {
      new_node.physical_index_position[new_node.num_active_indices++] = i;
    }
  }
  // infer extractors
  int acc_shape = 1;
  for (int i = taichi_max_num_indices - 1; i >= 0; i--) {
    new_node.extractors[i].acc_shape = acc_shape;
    acc_shape *= new_node.extractors[i].shape;
  }
  new_node.num_cells_per_container = acc_shape;
  // infer extractors (only for POT)
  int acc_offsets = 0;
  for (int i = taichi_max_num_indices - 1; i >= 0; i--) {
    new_node.extractors[i].acc_offset = acc_offsets;
    acc_offsets += new_node.extractors[i].num_bits;
  }
  new_node.total_num_bits = acc_offsets;

  constexpr int kMaxTotalNumBits = 64;
  TI_ERROR_IF(
      new_node.total_num_bits >= kMaxTotalNumBits,
      "SNode={}: total_num_bits={} exceeded limit={}. This implies that "
      "your requested shape is too large.",
      new_node.id, new_node.total_num_bits, kMaxTotalNumBits);

  if (new_node.type == SNodeType::dynamic) {
    int active_extractor_counder = 0;
    for (int i = 0; i < taichi_max_num_indices; i++) {
      if (new_node.extractors[i].num_bits != 0) {
        active_extractor_counder += 1;
        SNode *p = new_node.parent;
        while (p) {
          TI_ASSERT_INFO(
              p->extractors[i].num_bits == 0,
              "Dynamic SNode must have a standalone dimensionality.");
          p = p->parent;
        }
      }
    }
    TI_ASSERT_INFO(active_extractor_counder == 1,
                   "Dynamic SNode can have only one index extractor.");
  }
  return new_node;
}

SNode &SNode::dynamic(const Axis &expr, int n, int chunk_size, bool packed) {
  auto &snode = create_node({expr}, {n}, SNodeType::dynamic, packed);
  snode.chunk_size = chunk_size;
  return snode;
}

SNode &SNode::bit_struct(int num_bits, bool packed) {
  auto &snode = create_node({}, {}, SNodeType::bit_struct, packed);
  snode.physical_type =
      TypeFactory::get_instance().get_primitive_int_type(num_bits, false);
  return snode;
}

SNode &SNode::bit_array(const std::vector<Axis> &axes,
                        const std::vector<int> &sizes,
                        int bits,
                        bool packed) {
  auto &snode = create_node(axes, sizes, SNodeType::bit_array, packed);
  snode.physical_type =
      TypeFactory::get_instance().get_primitive_int_type(bits, false);
  return snode;
}

bool SNode::is_place() const {
  return type == SNodeType::place;
}

bool SNode::is_scalar() const {
  return is_place() && (num_active_indices == 0);
}

SNode *SNode::get_least_sparse_ancestor() const {
  if (is_path_all_dense) {
    return nullptr;
  }
  auto *result = const_cast<SNode *>(this);
  while (!result->need_activation()) {
    result = result->parent;
    TI_ASSERT(result);
  }
  return result;
}

int SNode::shape_along_axis(int i) const {
  const auto &extractor = extractors[physical_index_position[i]];
  return extractor.num_elements_from_root;
}

SNode::SNode() : SNode(0, SNodeType::undefined) {
}

SNode::SNode(int depth, SNodeType t) : depth(depth), type(t) {
  id = counter++;
  node_type_name = get_node_type_name();
  total_num_bits = 0;
  total_bit_start = 0;
  num_active_indices = 0;
  std::memset(physical_index_position, -1, sizeof(physical_index_position));
  parent = nullptr;
  has_ambient = false;
  dt = PrimitiveType::gen;
  _morton = false;
}

SNode::SNode(const SNode &) {
  TI_NOT_IMPLEMENTED;  // Copying an SNode is forbidden. However we need the
                       // definition here to make pybind11 happy.
}

std::string SNode::get_node_type_name() const {
  return fmt::format("S{}", id);
}

std::string SNode::get_node_type_name_hinted() const {
  std::string suffix;
  if (type == SNodeType::place || type == SNodeType::bit_struct ||
      type == SNodeType::bit_array)
    suffix = fmt::format("<{}>", dt->to_string());
  if (is_bit_level)
    suffix += "<bit>";
  return fmt::format("S{}{}{}", id, snode_type_name(type), suffix);
}

int SNode::get_num_bits(int physical_index) const {
  int result = 0;
  const SNode *snode = this;
  while (snode) {
    result += snode->extractors[physical_index].num_bits;
    snode = snode->parent;
  }
  return result;
}

void SNode::print() {
  for (int i = 0; i < depth; i++) {
    fmt::print("  ");
  }
  fmt::print("{}", get_node_type_name_hinted());
  if (exp_snode) {
    fmt::print(" exp={}", exp_snode->get_node_type_name());
  }
  fmt::print("\n");
  for (auto &c : ch) {
    c->print();
  }
}

void SNode::set_index_offsets(std::vector<int> index_offsets_) {
  TI_ASSERT(this->index_offsets.empty());
  TI_ASSERT(!index_offsets_.empty());
  TI_ASSERT(type == SNodeType::place);
  TI_ASSERT(index_offsets_.size() == this->num_active_indices);
  this->index_offsets = index_offsets_;
}

// TODO: rename to is_sparse?
bool SNode::need_activation() const {
  return type == SNodeType::pointer || type == SNodeType::hash ||
         type == SNodeType::bitmasked || type == SNodeType::dynamic;
}

void SNode::begin_shared_exp_placement() {
  TI_ASSERT(!placing_shared_exp);
  TI_ASSERT(currently_placing_exp_snode == nullptr);
  placing_shared_exp = true;
}

void SNode::end_shared_exp_placement() {
  TI_ASSERT(placing_shared_exp);
  TI_ASSERT(currently_placing_exp_snode != nullptr);
  currently_placing_exp_snode = nullptr;
  placing_shared_exp = false;
}

bool SNode::is_primal() const {
  return grad_info->is_primal();
}

bool SNode::has_grad() const {
  return is_primal() && (grad_info->grad_snode() != nullptr);
}

SNode *SNode::get_grad() const {
  TI_ASSERT(has_grad());
  return grad_info->grad_snode();
}

void SNode::set_snode_tree_id(int id) {
  snode_tree_id_ = id;
}

int SNode::get_snode_tree_id() {
  return snode_tree_id_;
}

TLANG_NAMESPACE_END
