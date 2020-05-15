#pragma once

#include <atomic>

#include "taichi/lang_util.h"
#include "taichi/util/bit.h"
#include "taichi/llvm/llvm_fwd.h"
#include "taichi/ir/expr.h"
#include "taichi/inc/constants.h"

TLANG_NAMESPACE_BEGIN

class Expr;
class Kernel;

struct IndexExtractor {
  int start;
  int num_bits;
  int acc_offset;
  int num_elements;

  // TODO: rename start to src_offset

  bool active;

  IndexExtractor() {
    start = 0;
    num_bits = 0;
    active = false;
    acc_offset = 0;
    num_elements = 1;
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
  int64 n{};
  int total_num_bits{}, total_bit_start{};
  int chunk_size{};
  DataType dt;
  bool has_ambient{};
  TypedConstant ambient_val;
  // Note: parent will not be set until structural nodes are compiled!
  SNode *parent{};
  Kernel *reader_kernel{};
  Kernel *writer_kernel{};
  Expr expr;

  SNode();

  SNode(int depth, SNodeType t);

  SNode(const SNode &);

  ~SNode() = default;

  std::string node_type_name;
  SNodeType type;
  bool _morton{};

  std::string get_node_type_name() const;

  std::string get_node_type_name_hinted() const;

  SNode &insert_children(SNodeType t) {
    ch.push_back(create(depth + 1, t));
    // Note: parent will not be set until structural nodes are compiled!
    return *ch.back();
  }

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

  template <typename... Args>
  static std::unique_ptr<SNode> create(Args &&... args) {
    return std::make_unique<SNode>(std::forward<Args>(args)...);
  }

  std::string type_name() {
    return snode_type_name(type);
  }

  void print();

  void set_index_offsets(std::vector<int> index_offsets);

  void place(Expr &expr, const std::vector<int> &offset);

  SNode &dynamic(const Index &expr, int n, int chunk_size);

  SNode &morton(bool val = true) {
    _morton = val;
    return *this;
  }

  // for float and double
  void write_float(const std::vector<int> &I, float64);
  float64 read_float(const std::vector<int> &I);

  // for int32 and int64
  void write_int(const std::vector<int> &I, int64);
  int64 read_int(const std::vector<int> &I);
  uint64 read_uint(const std::vector<int> &I);

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

  void lazy_grad();

  bool is_primal() const;

  bool is_place() const;

  const Expr &get_expr() const {
    return expr;
  }

  bool has_grad() const;

  SNode *get_grad() const;

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

  int max_num_elements() const {
    return 1 << total_num_bits;
  }

  int num_elements_along_axis(int i) const;

  void set_kernel_args(Kernel *kernel, const std::vector<int> &I);

  uint64 fetch_reader_result();  // TODO: refactor
};

TLANG_NAMESPACE_END
