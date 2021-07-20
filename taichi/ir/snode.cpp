#include "taichi/ir/snode.h"

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

std::atomic<int> SNode::counter{0};

SNode &SNode::insert_children(SNodeType t) {
  TI_ASSERT(t != SNodeType::root);

  auto new_ch = std::make_unique<SNode>(depth + 1, t);
  new_ch->is_path_all_dense = (is_path_all_dense && !new_ch->need_activation());
  ch.push_back(std::move(new_ch));

  // Note: |new_ch->parent| will not be set (or well-defined) until structural
  // nodes are compiled! This is because the structure compiler may modify the
  // SNode tree during compilation.

  return *ch.back();
}

SNode &SNode::create_node(std::vector<Index> indices,
                          std::vector<int> sizes,
                          SNodeType type) {
  TI_ASSERT(indices.size() == sizes.size() || sizes.size() == 1);
  if (sizes.size() == 1) {
    sizes = std::vector<int>(indices.size(), sizes[0]);
  }

  if (type == SNodeType::hash)
    TI_ASSERT_INFO(depth == 0,
                   "hashed node must be child of root due to initialization "
                   "memset limitation.");
  auto &new_node = insert_children(type);
  for (int i = 0; i < (int)indices.size(); i++) {
    TI_ASSERT(sizes[i] > 0);
    auto &ind = indices[i];
    new_node.extractors[ind.value].activate(
        bit::log2int(bit::least_pot_bound(sizes[i])));
    new_node.extractors[ind.value].shape = sizes[i];
    new_node.extractors[ind.value].num_elements_from_root = sizes[i];
  }
  return new_node;
}

SNode &SNode::dynamic(const Index &expr, int n, int chunk_size) {
  auto &snode = create_node({expr}, {n}, SNodeType::dynamic);
  snode.chunk_size = chunk_size;
  return snode;
}

SNode &SNode::bit_struct(int num_bits) {
  auto &snode = create_node({}, {}, SNodeType::bit_struct);
  snode.physical_type =
      TypeFactory::get_instance().get_primitive_int_type(num_bits, false);
  return snode;
}

SNode &SNode::bit_array(const std::vector<Index> &indices,
                        const std::vector<int> &sizes,
                        int bits) {
  auto &snode = create_node(indices, sizes, SNodeType::bit_array);
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
