#include "tlang.h"
#include "structural_node.h"
#include <tuple>

TLANG_NAMESPACE_BEGIN

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
  return (*this)[ExprGroup(i)];
}

Expr Expr::operator[](const ExprGroup &_is) {
  auto is = _is;
  TC_ASSERT(node->type == NodeType::addr);
  for (auto &i: is.exprs) {
    if (i->type == NodeType::pointer) {
      i.set(load(i));
    }
  }
  for (auto &i : is.exprs) {
    TC_ASSERT(i);
    TC_ASSERT(i->type == NodeType::index || i->data_type == DataType::i32);
  }
  if (is.size() == 0) {
    auto n = create(NodeType::pointer, *this);
    n->data_type = (*this)->data_type;
    return n;
  }else if (is.size() == 1) {
    auto n = create(NodeType::pointer, *this, is.exprs[0]);
    n->data_type = (*this)->data_type;
    return n;
  } else if (is.size() == 2) {
    auto n = create(NodeType::pointer, *this, is.exprs[0], is.exprs[1]);
    n->data_type = (*this)->data_type;
    return n;
  } else if (is.size() == 3) {
    auto n =
        create(NodeType::pointer, *this, is.exprs[0], is.exprs[1], is.exprs[2]);
    n->data_type = (*this)->data_type;
    return n;
  } else if (is.size() == 4) {
    auto n = create(NodeType::pointer, *this, is.exprs[0], is.exprs[1],
                    is.exprs[2], is.exprs[3]);
    n->data_type = (*this)->data_type;
    return n;
  } else {
    TC_NOT_IMPLEMENTED
  }
  TC_NOT_IMPLEMENTED
  return Expr();
}

void *Expr::evaluate_addr(int i, int j, int k, int l) {
  TC_ASSERT(node->lanes == 1);
  return node->snode_ptr(0)->evaluate(get_current_program().data_structure, i,
                                      j, k, l);
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
  SNode *snode = node->snode_ptr(0);
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

template void *Expr::val_tmp<>();
template void *Expr::val_tmp<int>(int);
template void *Expr::val_tmp<int, int>(int, int);
template void *Expr::val_tmp<int, int, int>(int, int, int);
template void *Expr::val_tmp<int, int, int, int>(int, int, int, int);

Expr Expr::index(int i) {
  if (i == -1) {
    i = get_current_program().index_counter++;
  }
  TC_ASSERT(i < max_num_indices);
  auto e = create(NodeType::index);
  e->value<int>() = i;
  e->data_type = DataType::i32;
  return e;
}

// sort all the stores
bool prior_to(const Expr &a, const Expr &b) {
  TC_ASSERT(a->lanes == 1 && b->lanes == 1);
  TC_ASSERT(a->type == NodeType::pointer && b->type == NodeType::pointer);
  auto sa = a._address()->snode_ptr(0), sb = b._address()->snode_ptr(0);
  if (sa->parent && sb->parent) {
    // TC_P(sa->parent->child_id(sa));
    // TC_P(sb->parent->child_id(sb));
    return sa->parent->child_id(sa) + 1 == sb->parent->child_id(sb);
  } else {
    return false;
  }
}

TLANG_NAMESPACE_END
