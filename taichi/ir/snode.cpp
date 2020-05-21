#include "taichi/ir/snode.h"

#include "taichi/ir/ir.h"
#include "taichi/ir/frontend.h"
#include "taichi/backends/cuda/cuda_driver.h"

TLANG_NAMESPACE_BEGIN

std::atomic<int> SNode::counter = 0;

void SNode::place(Expr &expr_, const std::vector<int> &offset) {
  if (type == SNodeType::root) {  // never directly place to root
    this->dense(std::vector<Index>(), {}).place(expr_, offset);
  } else {
    TI_ASSERT(expr_.is<GlobalVariableExpression>());
    auto expr = expr_.cast<GlobalVariableExpression>();
    TI_ERROR_IF(expr->snode != nullptr, "This variable has been placed.");
    auto &child = insert_children(SNodeType::place);
    expr->set_snode(&child);
    child.name = expr->ident.raw_name();
    if (expr->has_ambient) {
      expr->snode->has_ambient = true;
      expr->snode->ambient_val = expr->ambient_value;
    }
    expr->snode->expr.set(Expr(expr));
    child.dt = expr->dt;
    if (!offset.empty())
      child.set_index_offsets(offset);
  }
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
  new_node.n = 1;
  for (int i = 0; i < sizes.size(); i++) {
    auto s = sizes[i];
    TI_ASSERT(sizes[i] > 0);
    if (!bit::is_power_of_two(s)) {
      auto promoted_s = bit::least_pot_bound(s);
      TI_DEBUG("Non-power-of-two node size {} promoted to {}.", s, promoted_s);
      s = promoted_s;
    }
    TI_ASSERT(bit::is_power_of_two(s));
    new_node.n *= s;
  }
  for (int i = 0; i < (int)indices.size(); i++) {
    auto &ind = indices[i];
    new_node.extractors[ind.value].activate(
        bit::log2int(bit::least_pot_bound(sizes[i])));
    new_node.extractors[ind.value].num_elements = sizes[i];
  }
  return new_node;
}

SNode &SNode::dynamic(const Index &expr, int n, int chunk_size) {
  auto &snode = create_node({expr}, {n}, SNodeType::dynamic);
  snode.chunk_size = chunk_size;
  return snode;
}

void SNode::lazy_grad() {
  if (this->type == SNodeType::place)
    return;
  for (auto &c : ch) {
    c->lazy_grad();
  }
  std::vector<Expr> new_grads;
  for (auto &c : ch) {
    if (c->type == SNodeType::place && c->is_primal() && needs_grad(c->dt) &&
        !c->has_grad()) {
      new_grads.push_back(c->expr.cast<GlobalVariableExpression>()->adjoint);
    }
  }
  for (auto p : new_grads) {
    this->place(p, {});
  }
}

bool SNode::is_primal() const {
  TI_ASSERT(expr.expr != nullptr);
  return expr.cast<GlobalVariableExpression>()->is_primal;
}

bool SNode::is_place() const {
  return type == SNodeType::place;
}

bool SNode::has_grad() const {
  auto adjoint = expr.cast<GlobalVariableExpression>()->adjoint;
  return is_primal() && adjoint.expr != nullptr &&
         adjoint.cast<GlobalVariableExpression>()->snode != nullptr;
}

SNode *SNode::get_grad() const {
  TI_ASSERT(has_grad());
  return expr.cast<GlobalVariableExpression>()
      ->adjoint.cast<GlobalVariableExpression>()
      ->snode;
}

// for float and double
void SNode::write_float(const std::vector<int> &I, float64 val) {
  if (writer_kernel == nullptr) {
    writer_kernel = &get_current_program().get_snode_writer(this);
  }
  set_kernel_args(writer_kernel, I);
  for (int i = 0; i < num_active_indices; i++) {
    writer_kernel->set_arg_int(i, I[i]);
  }
  writer_kernel->set_arg_float(num_active_indices, val);
  get_current_program().synchronize();
  (*writer_kernel)();
}

float64 SNode::read_float(const std::vector<int> &I) {
  if (reader_kernel == nullptr) {
    reader_kernel = &get_current_program().get_snode_reader(this);
  }
  set_kernel_args(reader_kernel, I);
  get_current_program().synchronize();
  (*reader_kernel)();
  get_current_program().synchronize();
  auto ret = reader_kernel->get_ret_float(0);
  return ret;
}

// for int32 and int64
void SNode::write_int(const std::vector<int> &I, int64 val) {
  if (writer_kernel == nullptr) {
    writer_kernel = &get_current_program().get_snode_writer(this);
  }
  set_kernel_args(writer_kernel, I);
  writer_kernel->set_arg_int(num_active_indices, val);
  get_current_program().synchronize();
  (*writer_kernel)();
}

int64 SNode::read_int(const std::vector<int> &I) {
  if (reader_kernel == nullptr) {
    reader_kernel = &get_current_program().get_snode_reader(this);
  }
  set_kernel_args(reader_kernel, I);
  get_current_program().synchronize();
  (*reader_kernel)();
  get_current_program().synchronize();
  auto ret = reader_kernel->get_ret_int(0);
  return ret;
}

uint64 SNode::read_uint(const std::vector<int> &I) {
  return (uint64)read_int(I);
}

int SNode::num_elements_along_axis(int i) const {
  return extractors[physical_index_position[i]].num_elements;
}

void SNode::set_kernel_args(Kernel *kernel, const std::vector<int> &I) {
  for (int i = 0; i < num_active_indices; i++) {
    kernel->set_arg_int(i, I[i]);
  }
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
  dt = DataType::gen;
  _morton = false;

  reader_kernel = nullptr;
  writer_kernel = nullptr;
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
  if (type == SNodeType::place)
    suffix = fmt::format("_{}", data_type_short_name(dt));
  return fmt::format("S{}{}{}", id, snode_type_name(type), suffix);
}

void SNode::print() {
  for (int i = 0; i < depth; i++) {
    fmt::print("  ");
  }
  fmt::print("{}\n", get_node_type_name_hinted());
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

TLANG_NAMESPACE_END
