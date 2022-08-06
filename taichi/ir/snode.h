#pragma once

#include <atomic>

#include "taichi/inc/constants.h"
#include "taichi/ir/expr.h"
#include "taichi/ir/snode_types.h"
#include "taichi/ir/type.h"
#include "taichi/program/snode_expr_utils.h"

namespace taichi {
namespace lang {
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
  Axis(int value) : value(value) {
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
   *
   * This is the raw number, *not* padded to power-of-two (POT).
   */
  int num_elements_from_root{1};
  /**
   * Shape at this index (POT or packed) according to the config.
   */
  int shape{1};
  /**
   * Accumulated shape from the last activated index to the first one.
   */
  int acc_shape{1};
  /**
   * Number of bits needed to store the coordinate at this index.
   *
   * ceil(log2(shape))
   */
  int num_bits{0};
  /**
   * Accumulated offset from the last activated index to the first one.
   *
   * This is the starting bit of this index in a linearized 1D coordinate. For
   * example, assuming an SNode of (ti.ijk, shape=(4, 8, 16)). ti.i takes 2
   * bits, ti.j 3 bits and ti.k 4 bits. Then for a linearized coordinate:
   * ti.k uses bits [0, 4), acc_offset=0
   * ti.j uses bits [4, 7), acc_offset=4
   * ti.i uses bits [7, 9), acc_offset=7
   */
  int acc_offset{0};
  /**
   * Whether this index (axis) is activated.
   */
  bool active{false};

  /**
   * Activates the current index.
   *
   * @param num_bits Number of bits needed to store the POT shape.
   */
  void activate(int num_bits) {
    active = true;
    this->num_bits = num_bits;
  }
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
    virtual SNode *adjoint_snode() const = 0;
    virtual SNode *dual_snode() const = 0;

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
  int total_num_bits{0};
  int total_bit_start{0};
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

  SNode(SNodeGlobalVarExprMap *snode_to_glb_var_exprs = nullptr,
        SNodeRwAccessorsBank *snode_rw_accessors_bank = nullptr);

  SNode(int depth,
        SNodeType t,
        SNodeGlobalVarExprMap *snode_to_glb_var_exprs = nullptr,
        SNodeRwAccessorsBank *snode_rw_accessors_bank = nullptr);

  SNode(const SNode &);

  ~SNode() = default;

  std::string node_type_name;
  SNodeType type;
  bool _morton{false};

  std::string get_node_type_name() const;

  std::string get_node_type_name_hinted() const;

  int get_num_bits(int physical_index) const;

  SNode &insert_children(SNodeType t);

  SNode &create_node(std::vector<Axis> axes,
                     std::vector<int> sizes,
                     SNodeType type,
                     bool packed);

  // SNodes maintains how flattened index bits are taken from indices
  SNode &dense(const std::vector<Axis> &axes,
               const std::vector<int> &sizes,
               bool packed) {
    return create_node(axes, sizes, SNodeType::dense, packed);
  }

  SNode &dense(const std::vector<Axis> &axes, int sizes, bool packed) {
    return create_node(axes, std::vector<int>{sizes}, SNodeType::dense, packed);
  }

  SNode &dense(const Axis &axis, int size, bool packed) {
    return SNode::dense(std::vector<Axis>{axis}, size, packed);
  }

  SNode &pointer(const std::vector<Axis> &axes,
                 const std::vector<int> &sizes,
                 bool packed) {
    return create_node(axes, sizes, SNodeType::pointer, packed);
  }

  SNode &pointer(const std::vector<Axis> &axes, int sizes, bool packed) {
    return create_node(axes, std::vector<int>{sizes}, SNodeType::pointer,
                       packed);
  }

  SNode &pointer(const Axis &axis, int size, bool packed) {
    return SNode::pointer(std::vector<Axis>{axis}, size, packed);
  }

  SNode &bitmasked(const std::vector<Axis> &axes,
                   const std::vector<int> &sizes,
                   bool packed) {
    return create_node(axes, sizes, SNodeType::bitmasked, packed);
  }

  SNode &bitmasked(const std::vector<Axis> &axes, int sizes, bool packed) {
    return create_node(axes, std::vector<int>{sizes}, SNodeType::bitmasked,
                       packed);
  }

  SNode &bitmasked(const Axis &axis, int size, bool packed) {
    return SNode::bitmasked(std::vector<Axis>{axis}, size, packed);
  }

  SNode &hash(const std::vector<Axis> &axes,
              const std::vector<int> &sizes,
              bool packed) {
    return create_node(axes, sizes, SNodeType::hash, packed);
  }

  SNode &hash(const std::vector<Axis> &axes, int sizes, bool packed) {
    return create_node(axes, std::vector<int>{sizes}, SNodeType::hash, packed);
  }

  SNode &hash(const Axis &axis, int size, bool packed) {
    return hash(std::vector<Axis>{axis}, size, packed);
  }

  std::string type_name() {
    return snode_type_name(type);
  }

  SNode &bit_struct(BitStructType *bit_struct_type, bool packed);

  SNode &quant_array(const std::vector<Axis> &axes,
                     const std::vector<int> &sizes,
                     int bits,
                     bool packed);

  void print();

  void set_index_offsets(std::vector<int> index_offsets);

  SNode &dynamic(const Axis &expr, int n, int chunk_size, bool packed);

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

  bool has_adjoint() const;

  SNode *get_adjoint() const;

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
    place_child(&expr, offset, id_in_bit_struct, this, snode_to_glb_var_exprs_);
  }

  void lazy_grad(bool is_adjoint, bool is_dual) {
    make_lazy_grad(this, snode_to_glb_var_exprs_, is_adjoint, is_dual);
  }

  int64 read_int(const std::vector<int> &i);
  uint64 read_uint(const std::vector<int> &i);
  float64 read_float(const std::vector<int> &i);
  void write_int(const std::vector<int> &i, int64 val);
  void write_float(const std::vector<int> &i, float64 val);

  Expr get_expr() const;

  uint64 fetch_reader_result();  // TODO: refactor

  // SNodeTree part

  void set_snode_tree_id(int id);

  int get_snode_tree_id() const;

  static void reset_counter() {
    counter = 0;
  }

 private:
  int snode_tree_id_{0};
  SNodeGlobalVarExprMap *snode_to_glb_var_exprs_{nullptr};
  SNodeRwAccessorsBank *snode_rw_accessors_bank_{nullptr};
};

}  // namespace lang
}  // namespace taichi
