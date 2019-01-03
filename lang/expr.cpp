#include "tlang.h"
#include "structural_node.h"
#include <tuple>

TLANG_NAMESPACE_BEGIN

int Expr::index_counter = 0;

Expr &Expr::operator=(const Expr &o) {
  // TC_ASSERT(allow_store);
  if (!allow_store || !node || node->type != NodeType::pointer) {
    // Expr assignment
    node = o.node;
  } else {
    // store to pointed addr
    TC_ASSERT(node->type == NodeType::pointer);
    auto &prog = get_current_program();
    // TC_ASSERT(&prog != nullptr);
    // TC_ASSERT(node->get_address().initialized());
    prog.store(*this, load_if_pointer(o));
  }
  return *this;
}

Expr Expr::operator[](const Expr &i) {
  TC_ASSERT(i);
  TC_ASSERT(node->type == NodeType::addr);
  TC_ASSERT(i->type == NodeType::index || i->data_type == DataType::i32);
  return create(NodeType::pointer, *this, i);
}

Expr Expr::operator[](const ExprGroup &is) {
  TC_ASSERT(is.size() > 0 && is.size() <= 2);
  TC_ASSERT(node->type == NodeType::addr);
  for (auto &i : is.exprs) {
    TC_ASSERT(i);
    TC_ASSERT(i->type == NodeType::index || i->data_type == DataType::i32);
  }
  if (is.size() == 1) {
    auto n = create(NodeType::pointer, *this, is.exprs[0]);
    n->data_type = (*this)->data_type;
    return n;
  } else {
    auto n = create(NodeType::pointer, *this, is.exprs[0], is.exprs[1]);
    n->data_type = (*this)->data_type;
    return n;
  }
}

void *Expr::evaluate_addr(int i, int j, int k, int l) {
  TC_ASSERT(node->lanes == 1);
  return node->new_addresses(0)->evaluate(get_current_program().data_structure,
                                          i, j, k, l);
}

bool Expr::allow_store = false;
// assignment should not be used outside function definition; use "Expr::set"
// instead

template <int i, typename... Indices>
std::enable_if_t<(i < sizeof...(Indices)), int> get_if_exists(
    std::tuple<Indices...> tup) {
  return std::get<i>(tup);
}

template <int i, typename... Indices>
std::enable_if_t<!(i < sizeof...(Indices)), int> get_if_exists(
    std::tuple<Indices...> tup) {
  return 0;
}

template <typename... Indices>
void *Expr::val_tmp(Indices... indices) {
  TC_ASSERT(node->type == NodeType::addr);
  SNode *snode = node->new_addresses(0);
  TC_ASSERT(sizeof...(indices) == snode->num_active_indices);
  int ind[max_num_indices];
  std::memset(ind, 0, sizeof(ind));
  auto tup = std::make_tuple(indices...);
#define LOAD_IND(i) ind[snode->index_order[i]] = get_if_exists<i>(tup);
  LOAD_IND(0);
  LOAD_IND(1);
  LOAD_IND(2);
  LOAD_IND(3);
#undef LOAD_IND
  TC_ASSERT(max_num_indices == 4);
  /*
  TC_P(snode->index_order[0]);
  TC_P(snode->index_order[1]);
  TC_P(snode->index_order[2]);
  TC_P(snode->index_order[3]);
  TC_P(ind[0]);
  TC_P(ind[1]);
  TC_P(ind[2]);
  TC_P(ind[3]);
  TC_P(((int *)&tup)[0]);
  TC_P(((int *)&tup)[1]);
  TC_P(((int *)&tup)[2]);
  TC_P(((int *)&tup)[3]);
  */
  return evaluate_addr(ind[0], ind[1], ind[2], ind[3]);
}

template void *Expr::val_tmp<int>(int);
template void *Expr::val_tmp<int, int>(int, int);
template void *Expr::val_tmp<int, int, int>(int, int, int);
template void *Expr::val_tmp<int, int, int, int>(int, int, int, int);

TLANG_NAMESPACE_END
