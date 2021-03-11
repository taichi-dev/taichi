#pragma once

#include <atomic>

#include "taichi/inc/constants.h"
#include "taichi/ir/expr.h"
#include "taichi/ir/snode_types.h"
#include "taichi/ir/type.h"

TLANG_NAMESPACE_BEGIN

struct IndexExtractor {
  int start;
  int num_bits;
  int acc_offset;
  int num_elements;
  int trailing_bits;

  // TODO: rename start to src_offset

  bool active;

  IndexExtractor() {
    start = 0;
    num_bits = 0;
    active = false;
    acc_offset = 0;
    num_elements = 1;
    trailing_bits = 0;
  }

  void activate(int num_bits) {
    active = true;
    this->num_bits = num_bits;
  }
};

class Index {
 public:
  int value;
  Index() {
    value = 0;
  }
  Index(int value) : value(value) {
    TI_ERROR_UNLESS(0 <= value && value < taichi_max_num_indices,
                    "Too many dimensions. The maximum dimensionality is {}",
                    taichi_max_num_indices);
  }
};

// Structural nodes
class SNode {
 public:
  // This class decouples SNode from the frontend expression.
  class GradInfoProvider {
   public:
    virtual ~GradInfoProvider() = default;
    virtual bool is_primal() const = 0;
    virtual SNode *grad_snode() const = 0;

    template <typename T>
    T *cast() {
      return static_cast<T *>(this);
    }
  };
  std::vector<std::unique_ptr<SNode>> ch;

  IndexExtractor extractors[taichi_max_num_indices];
  std::vector<int> index_offsets;
  int num_active_indices{};
  int physical_index_position[taichi_max_num_indices]{};
  // physical indices are (ti.i, ti.j, ti.k, ti.l, ...)
  // physical_index_position[i] =
  // which physical index does the i-th virtual index (the one exposed to
  // programmers) refer to? i.e. in a[i, j, k], "i", "j", and "k" are virtual
  // indices.

  static std::atomic<int> counter;
  int id;
  int depth{};

  std::string name;
  int64 n{0};
  int total_num_bits{0};
  int total_bit_start{0};
  int chunk_size{0};
  std::size_t cell_size_bytes{0};
  PrimitiveType *physical_type;  // for bit_struct and bit_array only
  DataType dt;
  bool has_ambient{};
  TypedConstant ambient_val;
  // Note: parent will not be set until structural nodes are compiled!
  SNode *parent{};
  std::unique_ptr<GradInfoProvider> grad_info{nullptr};
  SNode *exp_snode{nullptr};  // for CustomFloatType with exponent bits
  int bit_offset{0};          // for children of bit_struct only
  bool placing_shared_exp{false};
  SNode *currently_placing_exp_snode{nullptr};
  Type *currently_placing_exp_snode_dtype{nullptr};
  bool owns_shared_exponent{false};
  std::vector<SNode *> exponent_users;

  // is_bit_level=false: the SNode is not bitpacked
  // is_bit_level=true: the SNode is bitpacked (i.e., strictly inside bit_struct
  // or bit_array)
  bool is_bit_level{false};

  // Whether the path from root to |this| contains only `dense` SNodes.
  bool is_path_all_dense{false};

  SNode();

  SNode(int depth, SNodeType t);

  SNode(const SNode &);

  ~SNode() = default;

  std::string node_type_name;
  SNodeType type;
  bool _morton{};

  std::string get_node_type_name() const;

  std::string get_node_type_name_hinted() const;

  int get_num_bits(int physical_index) const;

  SNode &insert_children(SNodeType t);

  SNode &create_node(std::vector<Index> indices,
                     std::vector<int> sizes,
                     SNodeType type);

  // SNodes maintains how flattened index bits are taken from indices
  SNode &dense(const std::vector<Index> &indices,
               const std::vector<int> &sizes) {
    return create_node(indices, sizes, SNodeType::dense);
  }

  SNode &dense(const std::vector<Index> &indices, int sizes) {
    return create_node(indices, std::vector<int>{sizes}, SNodeType::dense);
  }

  SNode &dense(const Index &index, int size) {
    return SNode::dense(std::vector<Index>{index}, size);
  }

  SNode &pointer(const std::vector<Index> &indices,
                 const std::vector<int> &sizes) {
    return create_node(indices, sizes, SNodeType::pointer);
  }

  SNode &pointer(const std::vector<Index> &indices, int sizes) {
    return create_node(indices, std::vector<int>{sizes}, SNodeType::pointer);
  }

  SNode &pointer(const Index &index, int size) {
    return SNode::pointer(std::vector<Index>{index}, size);
  }

  SNode &bitmasked(const std::vector<Index> &indices,
                   const std::vector<int> &sizes) {
    return create_node(indices, sizes, SNodeType::bitmasked);
  }

  SNode &bitmasked(const std::vector<Index> &indices, int sizes) {
    return create_node(indices, std::vector<int>{sizes}, SNodeType::bitmasked);
  }

  SNode &bitmasked(const Index &index, int size) {
    return SNode::bitmasked(std::vector<Index>{index}, size);
  }

  SNode &hash(const std::vector<Index> &indices,
              const std::vector<int> &sizes) {
    return create_node(indices, sizes, SNodeType::hash);
  }

  SNode &hash(const std::vector<Index> &indices, int sizes) {
    return create_node(indices, std::vector<int>{sizes}, SNodeType::hash);
  }

  SNode &hash(const Index &index, int size) {
    return hash(std::vector<Index>{index}, size);
  }

  std::string type_name() {
    return snode_type_name(type);
  }

  SNode &bit_struct(int bits);

  SNode &bit_array(const std::vector<Index> &indices,
                   const std::vector<int> &sizes,
                   int bits);

  void print();

  void set_index_offsets(std::vector<int> index_offsets);

  SNode &dynamic(const Index &expr, int n, int chunk_size);

  SNode &morton(bool val = true) {
    _morton = val;
    return *this;
  }

  int child_id(SNode *c) {
    for (int i = 0; i < (int)ch.size(); i++) {
      if (ch[i].get() == c) {
        return i;
      }
    }
    return -1;
  }

  bool has_null() const {
    return type == SNodeType::pointer || type == SNodeType::hash;
  }

  bool has_allocator() const {
    return type == SNodeType::pointer || type == SNodeType::hash ||
           type == SNodeType::root;
  }

  bool need_activation() const;

  bool is_primal() const;

  bool is_place() const;

  bool is_scalar() const;

  bool has_grad() const;

  SNode *get_grad() const;

  SNode *get_least_sparse_ancestor() const;

  std::string get_name() const {
    return node_type_name;
  }

  std::string element_listgen_func_name() const {
    return get_name() + "_element_listgen";
  }

  std::string get_ch_from_parent_func_name() const {
    TI_ASSERT(parent != nullptr);
    return fmt::format("get_ch_{}_to_{}", parent->get_name(), get_name());
  }

  std::string refine_coordinates_func_name() const {
    TI_ASSERT(type != SNodeType::place);
    return fmt::format("{}_refine_coordinates", get_name());
  }

  int64 max_num_elements() const {
    return int64(1) << total_num_bits;
  }

  int shape_along_axis(int i) const;

  uint64 fetch_reader_result();  // TODO: refactor

  void begin_shared_exp_placement();

  void end_shared_exp_placement();
};

TLANG_NAMESPACE_END
