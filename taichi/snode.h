#pragma once
#include "tlang_util.h"
#include "llvm_fwd.h"
#include "expr.h"
#include <taichi/common/bit.h>

TLANG_NAMESPACE_BEGIN

struct Matrix;
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
    TC_ERROR_UNLESS(0 <= value && value < max_num_indices,
                    "Too many dimensions. The maximum dimensionality is {}",
                    max_num_indices);
  }
};

// "Structural" nodes
class SNode {
 public:
  // Children
  std::vector<Handle<SNode>> ch;

  IndexExtractor extractors[max_num_indices];
  int taken_bits[max_num_indices]{};  // counting from the tail
  int num_active_indices{};
  int physical_index_position[max_num_indices]{};
  // physical indices are (ti.i, ti.j, ti.k, ti.l, ...)
  // physical_index_position[i] =
  // which physical index does the i-th virtual index (the one exposed to
  // programmers) refer to? i.e. in a[i, j, k], "i", "j", and "k" are virtual
  // indices.

  static int counter;
  int id;
  int depth{};
  bool _verbose{};
  bool _multi_threaded{};

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

  std::string data_type_name() {
    return Tlang::data_type_name(dt);
  }

  using AccessorFunction = std::function<void *(void *, int, int, int, int)>;
  using StatFunction = std::function<AllocatorStat()>;
  using ClearFunction = std::function<void(int)>;
  AccessorFunction access_func;
  StatFunction stat_func;
  ClearFunction clear_func;
  void *clear_kernel{}, *clear_and_deactivate_kernel{};

  std::string node_type_name;
  SNodeType type;
  int index_id{};
  bool _morton{};
  bool _bitmasked{};
  bool has_aux_structure{};

  std::string get_node_type_name() const {
    return fmt::format("S{}", id);
  }

  std::string get_node_type_name_hinted() const {
    return fmt::format("S{}{}", id, snode_type_name(type));
  }

  SNode();

  SNode(int depth, SNodeType t);

  ~SNode();

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

  SNode &multi_threaded(bool val = true) {
    this->_multi_threaded = val;
    return *this;
  }

  SNode &verbose() {
    this->_verbose = true;
    return *this;
  }

  template <typename... Args>
  SNode &place(Expr &expr, Args &&... args) {
    return place(expr).place(std::forward<Args>(args)...);
  }

  SNode &place(Matrix &mat);

  template <typename... Args>
  static Handle<SNode> create(Args &&... args) {
    return std::make_shared<SNode>(std::forward<Args>(args)...);
  }

  std::string type_name() {
    return snode_type_name(type);
  }

  void print() {
    for (int i = 0; i < depth; i++) {
      fmt::print("  ");
    }
    fmt::print("{}\n", type_name());
    for (auto c : ch) {
      c->print();
    }
  }

  SNode &place(Expr &expr);

  SNode &dynamic_chunked(const Index &expr, int n, int chunk_size) {
    TC_ASSERT(bit::is_power_of_two(n));
    TC_ASSERT(bit::is_power_of_two(chunk_size));
    auto &child = insert_children(SNodeType::dynamic);
    child.extractors[expr.value].activate(bit::log2int(n));
    child.n = n;
    child.chunk_size = chunk_size;
    return child;
  }

  SNode &hash(const std::vector<Index> indices, std::vector<int> sizes) {
    return create_node(indices, sizes, SNodeType::hash);
  }

  SNode &hash(const std::vector<Index> indices, int sizes) {
    return create_node(indices, std::vector<int>{sizes}, SNodeType::hash);
  }

  SNode &hash(const Index &expr, int n) {
    return hash(std::vector<Index>{expr}, n);
  }

  SNode &pointer() {
    return insert_children(SNodeType::pointer);
  }

  SNode &morton(bool val = true) {
    _morton = val;
    return *this;
  }

  SNode &bitmasked(bool val = true) {
    _bitmasked = val;
    return *this;
  }

  void *evaluate(void *ds, int i, int j, int k, int l) {
    TC_ASSERT(access_func);
    TC_ASSERT(max_num_indices == 4);
    return access_func(ds, i, j, k, l);
  }

  // for float and double
  void write_float(const std::vector<int> &I, float64);
  float64 read_float(const std::vector<int> &I);

  // for int32 and int64
  void write_int(const std::vector<int> &I, int64);
  int64 read_int(const std::vector<int> &I);

  TC_FORCE_INLINE AllocatorStat stat() {
    TC_ASSERT(stat_func);
    return stat_func();
  }

  int child_id(SNode *c) {
    for (int i = 0; i < (int)ch.size(); i++) {
      if (ch[i].get() == c) {
        return i;
      }
    }
    return -1;
  }

  void clear_data();

  void clear_data_and_deactivate();

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
    TC_ASSERT(parent != nullptr);
    return fmt::format("get_ch_{}_to_{}", parent->get_name(), get_name());
  }

  std::string refine_coordinates_func_name() const {
    TC_ASSERT(type != SNodeType::place);
    return fmt::format("{}_refine_coordinates", get_name());
  }

  int max_num_elements() const {
    return 1 << total_num_bits;
  }

  int num_elements_along_axis(int i) const;

  void set_kernel_args(Kernel *kernel, const std::vector<int> &I);
};

class SNodeAttribute {
 public:
  llvm::Type *llvm_type, *llvm_body_type, *llvm_aux_type;
  llvm::Type *llvm_element_type;
};

class SNodeAttributes {
 private:
  std::map<SNode *, SNodeAttribute> snode_llvm_attr;

 public:
  SNodeAttribute &operator[](SNode *snode) {
    return snode_llvm_attr[snode];
  }

  SNodeAttribute &operator[](SNode &snode) {
    return snode_llvm_attr[&snode];
  }

  SNodeAttribute &operator[](const Handle<SNode> &snode) {
    return snode_llvm_attr[snode.get()];
  }
};

TLANG_NAMESPACE_END
