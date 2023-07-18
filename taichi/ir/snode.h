#pragma once

#include <atomic>

#include "taichi/inc/constants.h"
#include "taichi/ir/expr.h"
#include "taichi/ir/snode_types.h"
#include "taichi/ir/type.h"
#include "taichi/program/snode_expr_utils.h"

namespace taichi::lang {
class Program;
class SNodeRwAccessorsBank;

/**
 * Dimension (or axis) of a tensor.
 *
 * For example, in the frontend we have ti.ij, which is translated to
 * {Axis{0}, Axis{1}}.
 */
class Axis {
 public:
  int value;
  Axis() {
    value = 0;
  }
  explicit Axis(int value) : value(value) {
    TI_ERROR_UNLESS(0 <= value && value < taichi_max_num_indices,
                    "Too many dimensions. The maximum dimensionality is {}",
                    taichi_max_num_indices);
  }
};

/**
 * SNode shape metadata at a specific Axis.
 */
struct AxisExtractor {
  /**
   * Number of elements from root at this index.
   */
  int num_elements_from_root{1};
  /**
   * Shape at this index.
   */
  int shape{1};
  /**
   * Accumulated shape from the last activated index to the first one.
   */
  int acc_shape{1};
  /**
   * Whether this index (axis) is activated.
   */
  bool active{false};
};

/**
 * Structural nodes
 */
class SNode {
 public:
  // This class decouples SNode from the frontend expression.
  class GradInfoProvider {
   public:
    virtual ~GradInfoProvider() = default;
    virtual bool is_primal() const = 0;
    virtual SNodeGradType get_snode_grad_type() const = 0;
    virtual SNode *adjoint_snode() const = 0;
    virtual SNode *dual_snode() const = 0;
    virtual SNode *adjoint_checkbit_snode() const = 0;

    template <typename T>
    T *cast() {
      return static_cast<T *>(this);
    }
  };
  std::vector<std::unique_ptr<SNode>> ch;

  AxisExtractor extractors[taichi_max_num_indices];
  std::vector<int> index_offsets;
  int num_active_indices{0};
  int physical_index_position[taichi_max_num_indices]{};
  // physical indices are (ti.i, ti.j, ti.k, ti.l, ...)
  // physical_index_position[i] =
  // which physical index does the i-th virtual index (the one exposed to
  // programmers) refer to? i.e. in a[i, j, k], "i", "j", and "k" are virtual
  // indices.

  static std::atomic<int> counter;
  int id{0};
  int depth{0};

  std::string name;
  // Product of the |shape| of all the activated axes identified by
  // |extractors|.
  // See https://docs.taichi-lang.org/docs/internal for terms
  // like cell and container.
  int64 num_cells_per_container{1};
  int chunk_size{0};
  std::size_t cell_size_bytes{0};
  std::size_t offset_bytes_in_parent_cell{0};
  DataType dt;
  bool has_ambient{false};
  TypedConstant ambient_val;
  // Note: parent will not be set until structural nodes are compiled!
  SNode *parent{nullptr};
  std::unique_ptr<GradInfoProvider> grad_info{nullptr};

  // Quant
  PrimitiveType *physical_type{nullptr};  // for bit_struct and quant_array only
  int id_in_bit_struct{-1};               // for children of bit_struct only
  bool is_bit_level{false};  // true if inside bit_struct or quant_array

  // Whether the path from root to |this| contains only `dense` SNodes.
  bool is_path_all_dense{true};

  explicit SNode(SNodeFieldMap *snode_to_fields = nullptr,
                 SNodeRwAccessorsBank *snode_rw_accessors_bank = nullptr);

  SNode(int depth,
        SNodeType t,
        SNodeFieldMap *snode_to_fields = nullptr,
        SNodeRwAccessorsBank *snode_rw_accessors_bank = nullptr);

  SNode(const SNode &);

  ~SNode() = default;

  std::string node_type_name;
  SNodeType type;
  bool _morton{false};

  std::string get_node_type_name() const;

  std::string get_node_type_name_hinted() const;

  SNode &insert_children(SNodeType t);

  SNode &create_node(std::vector<Axis> axes,
                     std::vector<int> sizes,
                     SNodeType type,
                     const DebugInfo &dbg_info = DebugInfo());

  // SNodes maintains how flattened index bits are taken from indices
  SNode &dense(const std::vector<Axis> &axes,
               const std::vector<int> &sizes,
               const DebugInfo &dbg_info = DebugInfo()) {
    return create_node(axes, sizes, SNodeType::dense, dbg_info);
  }

  SNode &dense(const std::vector<Axis> &axes,
               int sizes,
               const DebugInfo &dbg_info = DebugInfo()) {
    return create_node(axes, std::vector<int>{sizes}, SNodeType::dense,
                       dbg_info);
  }

  SNode &dense(const Axis &axis,
               int size,
               const DebugInfo &dbg_info = DebugInfo()) {
    return SNode::dense(std::vector<Axis>{axis}, size, dbg_info);
  }

  SNode &pointer(const std::vector<Axis> &axes,
                 const std::vector<int> &sizes,
                 const DebugInfo &dbg_info = DebugInfo()) {
    return create_node(axes, sizes, SNodeType::pointer, dbg_info);
  }

  SNode &pointer(const std::vector<Axis> &axes,
                 int sizes,
                 const DebugInfo &dbg_info = DebugInfo()) {
    return create_node(axes, std::vector<int>{sizes}, SNodeType::pointer,
                       dbg_info);
  }

  SNode &pointer(const Axis &axis,
                 int size,
                 const DebugInfo &dbg_info = DebugInfo()) {
    return SNode::pointer(std::vector<Axis>{axis}, size, dbg_info);
  }

  SNode &bitmasked(const std::vector<Axis> &axes,
                   const std::vector<int> &sizes,
                   const DebugInfo &dbg_info = DebugInfo()) {
    return create_node(axes, sizes, SNodeType::bitmasked, dbg_info);
  }

  SNode &bitmasked(const std::vector<Axis> &axes,
                   int sizes,
                   const DebugInfo &dbg_info = DebugInfo()) {
    return create_node(axes, std::vector<int>{sizes}, SNodeType::bitmasked,
                       dbg_info);
  }

  SNode &bitmasked(const Axis &axis,
                   int size,
                   const DebugInfo &dbg_info = DebugInfo()) {
    return SNode::bitmasked(std::vector<Axis>{axis}, size, dbg_info);
  }

  SNode &hash(const std::vector<Axis> &axes,
              const std::vector<int> &sizes,
              const DebugInfo &dbg_info = DebugInfo()) {
    return create_node(axes, sizes, SNodeType::hash, dbg_info);
  }

  SNode &hash(const std::vector<Axis> &axes,
              int sizes,
              const DebugInfo &dbg_info = DebugInfo()) {
    return create_node(axes, std::vector<int>{sizes}, SNodeType::hash,
                       dbg_info);
  }

  SNode &hash(const Axis &axis,
              int size,
              const DebugInfo &dbg_info = DebugInfo()) {
    return hash(std::vector<Axis>{axis}, size, dbg_info);
  }

  std::string type_name() {
    return snode_type_name(type);
  }

  SNode &bit_struct(BitStructType *bit_struct_type,
                    const DebugInfo &dbg_info = DebugInfo());

  SNode &quant_array(const std::vector<Axis> &axes,
                     const std::vector<int> &sizes,
                     int bits,
                     const DebugInfo &dbg_info = DebugInfo());

  void print();

  void set_index_offsets(std::vector<int> index_offsets);

  SNode &dynamic(const Axis &expr,
                 int n,
                 int chunk_size,
                 const DebugInfo &dbg_info = DebugInfo());

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

  SNodeGradType get_snode_grad_type() const;

  bool is_place() const;

  bool is_scalar() const;

  bool has_adjoint() const;

  SNode *get_adjoint() const;

  bool has_adjoint_checkbit() const;

  SNode *get_adjoint_checkbit() const;

  bool has_dual() const;

  SNode *get_dual() const;

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
    return num_cells_per_container;
  }

  int64 get_total_num_elements_towards_root() const {
    int64 total_num_elemts = 1;
    for (auto *s = this; s != nullptr; s = s->parent)
      total_num_elemts *= (int)s->max_num_elements();
    return total_num_elemts;
  }

  int shape_along_axis(int i) const;

  void place(Expr &expr, const std::vector<int> &offset, int id_in_bit_struct) {
    place_child(&expr, offset, id_in_bit_struct, this, snode_to_fields_);
  }

  void lazy_grad();

  void lazy_dual();

  void allocate_adjoint_checkbit();

  int64 read_int(const std::vector<int> &i);
  uint64 read_uint(const std::vector<int> &i);
  float64 read_float(const std::vector<int> &i);
  void write_int(const std::vector<int> &i, int64 val);
  void write_uint(const std::vector<int> &i, uint64 val);
  void write_float(const std::vector<int> &i, float64 val);

  Expr get_expr() const;

  uint64 fetch_reader_result();  // TODO: refactor

  // SNodeTree part

  void set_snode_tree_id(int id);

  int get_snode_tree_id() const;

  const SNode *get_root() const;

  static void reset_counter() {
    counter = 0;
  }

 private:
  int snode_tree_id_{0};
  SNodeFieldMap *snode_to_fields_{nullptr};
  SNodeRwAccessorsBank *snode_rw_accessors_bank_{
      nullptr};  // owned by the "Program" class
};

}  // namespace taichi::lang
