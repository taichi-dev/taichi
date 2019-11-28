#include "snode.h"
#include "ir.h"
#include "tlang.h"
// #include "math.h"

TLANG_NAMESPACE_BEGIN

int SNode::counter = 0;

SNode &SNode::place(Expr &expr_) {
  TC_ASSERT(expr_.is<GlobalVariableExpression>());
  auto expr = expr_.cast<GlobalVariableExpression>();
  TC_ERROR_UNLESS(expr->snode == nullptr, "This variable has been placed.");
  auto &child = insert_children(SNodeType::place);
  expr->set_snode(&child);
  child.name = expr->ident.raw_name();
  if (expr->has_ambient) {
    expr->snode->has_ambient = true;
    expr->snode->ambient_val = expr->ambient_value;
  }
  expr->snode->expr = std::make_unique<Expr>(expr);
  child.dt = expr->dt;
  return *this;
}

SNode &SNode::create_node(std::vector<Index> indices, std::vector<int> sizes,
                          SNodeType type) {
  TC_ASSERT(indices.size() == sizes.size() || sizes.size() == 1);
  if (sizes.size() == 1) {
    sizes = std::vector<int>(indices.size(), sizes[0]);
  }

  if (type == SNodeType::hash)
    TC_ASSERT_INFO(depth == 0,
                   "hashed node must be child of root due to initialization "
                   "memset limitation.");
  auto &new_node = insert_children(type);
  new_node.n = 1;
  for (int i = 0; i < sizes.size(); i++) {
    auto s = sizes[i];
    if (!bit::is_power_of_two(s)) {
      auto promoted_s = bit::least_pot_bound(s);
      TC_WARN("Non-power-of-two node size {} promoted to {}.", s, promoted_s);
      s = promoted_s;
    }
    TC_ASSERT(bit::is_power_of_two(s));
    new_node.n *= s;
  }
  for (int i = 0; i < (int)indices.size(); i++) {
    auto &ind = indices[i];
    new_node.extractors[ind.value].activate(bit::log2int(sizes[i]));
    new_node.extractors[ind.value].num_elements = sizes[i];
  }
  return new_node;
}

void SNode::clear_data() {
  if (clear_func == nullptr) {
    if (clear_kernel == nullptr) {
      clear_kernel = &kernel([&]() {
        current_ast_builder().insert(Stmt::make<ClearAllStmt>(this, false));
      });
    }
    (*(Kernel *)clear_kernel)();
  } else {
    clear_func(0);
  }
}

void SNode::clear_data_and_deactivate() {
  if (clear_func == nullptr) {
    if (clear_and_deactivate_kernel == nullptr) {
      clear_and_deactivate_kernel = &kernel([&]() {
        current_ast_builder().insert(Stmt::make<ClearAllStmt>(this, true));
      });
    }
    (*(Kernel *)clear_and_deactivate_kernel)();
  } else {
    clear_func(1);
  }
}

void SNode::lazy_grad() {
  if (this->type == SNodeType::place)
    return;
  for (auto c : ch) {
    c->lazy_grad();
  }
  std::vector<Expr> new_grads;
  for (auto c : ch) {
    if (c->type == SNodeType::place && c->is_primal() && needs_grad(c->dt) &&
        !c->has_grad()) {
      new_grads.push_back(c->expr->cast<GlobalVariableExpression>()->adjoint);
    }
  }
  for (auto p : new_grads) {
    this->place(p);
  }
}

bool SNode::is_primal() const {
  TC_ASSERT(expr != nullptr);
  return (*expr).cast<GlobalVariableExpression>()->is_primal;
}

bool SNode::has_grad() const {
  auto adjoint = (*expr).cast<GlobalVariableExpression>()->adjoint;
  return is_primal() && adjoint.expr != nullptr &&
         adjoint.cast<GlobalVariableExpression>()->snode != nullptr;
}

SNode *SNode::get_grad() const {
  TC_ASSERT(has_grad());
  return (*expr)
      .cast<GlobalVariableExpression>()
      ->adjoint.cast<GlobalVariableExpression>()
      ->snode;
}

// for float and double
void SNode::write_float(int i, int j, int k, int l, float64 val) {
  if (writer_kernel == nullptr) {
    writer_kernel = &get_current_program().get_snode_writer(this);
  }
  set_kernel_args(writer_kernel, i, j, k, l);
  if (num_active_indices >= 1)
    writer_kernel->set_arg_int(0, i);
  if (num_active_indices >= 2)
    writer_kernel->set_arg_int(1, j);
  if (num_active_indices >= 3)
    writer_kernel->set_arg_int(2, k);
  if (num_active_indices >= 4)
    writer_kernel->set_arg_int(3, l);
  writer_kernel->set_arg_float(num_active_indices, val);
  get_current_program().synchronize();
  (*writer_kernel)();
}

float64 SNode::read_float(int i, int j, int k, int l) {
  if (reader_kernel == nullptr) {
    reader_kernel = &get_current_program().get_snode_reader(this);
  }
  set_kernel_args(reader_kernel, i, j, k, l);
  get_current_program().synchronize();
  (*reader_kernel)();
  if (dt == DataType::f32) {
    return get_current_program().context.get_arg<float32>(num_active_indices);
  } else if (dt == DataType::f64) {
    return get_current_program().context.get_arg<float64>(num_active_indices);
  } else {
    TC_NOT_IMPLEMENTED
  }
}

// for int32 and int64
void SNode::write_int(int i, int j, int k, int l, int64 val) {
  if (writer_kernel == nullptr) {
    writer_kernel = &get_current_program().get_snode_writer(this);
  }
  set_kernel_args(writer_kernel, i, j, k, l);
  writer_kernel->set_arg_float(num_active_indices, val);
  get_current_program().synchronize();
  (*writer_kernel)();
}

int64 SNode::read_int(int i, int j, int k, int l) {
  if (reader_kernel == nullptr) {
    reader_kernel = &get_current_program().get_snode_reader(this);
  }
  set_kernel_args(reader_kernel, i, j, k, l);
  get_current_program().synchronize();
  (*reader_kernel)();
  if (dt == DataType::i32) {
    return get_current_program().context.get_arg<int32>(num_active_indices);
  } else if (dt == DataType::i64) {
    return get_current_program().context.get_arg<int64>(num_active_indices);
  } else {
    TC_NOT_IMPLEMENTED
  }
}

int SNode::num_elements_along_axis(int i) const {
  return extractors[physical_index_position[i]].num_elements;
}

void SNode::set_kernel_args(Kernel *kernel, int i, int j, int k, int l) {
  if (num_active_indices >= 1)
    kernel->set_arg_int(0, i);
  if (num_active_indices >= 2)
    kernel->set_arg_int(1, j);
  if (num_active_indices >= 3)
    kernel->set_arg_int(2, k);
  if (num_active_indices >= 4)
    kernel->set_arg_int(3, l);
}

TLANG_NAMESPACE_END