#include "vectorizer.h"

namespace taichi::Tlang {

void Vectorizer::sort(Expr &expr) {
  auto ch = expr->ch;  // a bunch of store nodes
  std::vector<Expr> sorted;

  while (!ch.empty()) {
    std::vector<Expr> group;
    group.push_back(ch[0]);
    ch.erase(ch.begin());
    while (true) {  // grow
      bool found = false;
      for (int i = 0; i < (int)ch.size(); i++) {  // search
        if (prior_to(ch[i]->addr(), group.front()->addr())) {
          group.insert(group.begin(), ch[i]);
          ch.erase(ch.begin() + i);
          found = true;
          break;
        }
        if (prior_to(group.back()->addr(), ch[i]->addr())) {
          group.insert(group.end(), ch[i]);
          ch.erase(ch.begin() + i);
          found = true;
          break;
        }
      }
      if (!found) {
        break;
      }
    }
    TC_ASSERT(group.size() % group_size == 0);
    sorted.insert(sorted.end(), group.begin(), group.end());
  }
  expr->ch = sorted;
}

Expr Vectorizer::run(Expr &expr, int group_size) {
  TC_ASSERT(expr);
  this->group_size = group_size;

  scalar_to_vector.clear();
  // expr should be a ret Op, with its children store Ops.
  // The stores are repeated by a factor of 'pack_size'
  // TC_P(group_size);
  TC_ASSERT(expr->ch.size() % group_size == 0);
  TC_ASSERT(expr->type == NodeType::combine);
  // Create the root group
  auto combined = Expr::create(NodeType::combine);
  combined->is_vectorized = true;

  TC_ASSERT(expr->ch.size());
  if (expr->ch[0]->type == NodeType::adapter_store) {
    // cache store
    // for each batch (group)
    for (int k = 0; k < (int)expr->ch.size() / group_size; k++) {
      auto root = Expr::create(NodeType::adapter_store);
      root->is_vectorized = true;
      for (int i = 0; i < group_size; i++) {
        TC_ASSERT(expr[k]->type == NodeType::adapter_store);
        auto ch = expr->ch[k * group_size + i];
        TC_ASSERT(ch->type == NodeType::adapter_store);
        root->members.push_back(ch);  // put scalar inst into vector members
        TC_ASSERT(i < (int)root->members.size());
      }
      root.accept(*this);
      combined->ch.push_back(root);
    }
  } else if (expr->ch[0]->type == NodeType::store) {
    // main memory store
    sort(expr);

    // for each batch (group)
    for (int k = 0; k < (int)expr->ch.size() / group_size; k++) {
      TC_ASSERT(expr[k]->type == NodeType::store);
      auto root = Expr::create(NodeType::store);
      root->is_vectorized = true;
      bool has_prior_to = false, has_same = false;
      for (int i = 0; i < group_size; i++) {
        auto ch = expr->ch[k * group_size + i];
        TC_ASSERT(ch->type == NodeType::store);
        root->members.push_back(ch);  // put scalar inst into vector members
        TC_ASSERT(i < (int)root->members.size());
        if (i > k * group_size) {
          if (prior_to(root->members[i - 1]->ch[0], root->members[i]->ch[0])) {
            has_prior_to = true;
          } else if (root->members[i - 1]->ch[0]->get_address() ==
                     root->members[i]->ch[0]->get_address()) {
            has_same = true;
          } else {
            TC_P(root->members[i - 1]->ch[0]->get_address());
            TC_P(root->members[i]->ch[0]->get_address());
            TC_ERROR(
                "Addresses in SIMD load should be either identical or "
                "neighbouring.");
          }
        }
      }
      TC_ASSERT(!(has_prior_to && has_same));
      // TC_P(root->members.size());
      root.accept(*this);
      combined->ch.push_back(root);
    }
  } else {
    TC_NOT_IMPLEMENTED
  }
  // TC_P(combined->ch.size());
  return combined;
}

void Vectorizer::visit(Expr &expr) {
  // TC_INFO("Visiting {} {}", expr->id, expr->node_type_name());
  // TC_P(expr->node_type_name());
  // Note: expr may be replaced by an existing vectorized Expr
  if (scalar_to_vector.find(expr->members[0]) != scalar_to_vector.end()) {
    auto existing = scalar_to_vector[expr->members[0]];
    TC_ASSERT(existing->members.size() == expr->members.size());
    bool mismatch = false;
    for (int i = 0; i < (int)existing->members.size(); i++) {
      if (existing->members[i] != expr->members[i])
        mismatch = true;
    }
    if (mismatch) {
      for (int i = 0; i < (int)existing->members.size(); i++) {
        TC_P(i);
        TC_P(existing->members[i]->id);
        TC_P(existing->members[i]->node_type_name());
        TC_P(expr->members[i]->id);
        TC_P(expr->members[i]->node_type_name());
        if (existing->members[i] != expr->members[i])
          mismatch = true;
        TC_WARN_UNLESS(existing->members[i] == expr->members[i], "mismatch");
      }
    }
    TC_ASSERT(!mismatch);
    expr.set(existing);
    // TC_WARN("Using existing {} for {}", existing->id, expr->id);
    return;
  }

  expr->is_vectorized = true;
  bool first = true;
  NodeType type;
  std::vector<std::vector<Expr>> vectorized_children;

  // Check for isomorphism
  for (auto member : expr->members) {
    // It must not appear to an existing vectorized expr
    // TC_ASSERT(scalar_to_vector.find(member) == scalar_to_vector.end());
    if (first) {
      first = false;
      type = member->type;
      vectorized_children.resize(member->ch.size());
    } else {
      TC_ASSERT(type == member->type);
      TC_ASSERT(vectorized_children.size() == member->ch.size());
    }
    for (int i = 0; i < (int)member->ch.size(); i++) {
      vectorized_children[i].push_back(member->ch[i]);
    }
    scalar_to_vector[member] = expr;
  }

  expr->is_vectorized = true;
  expr->data_type = expr->members[0]->data_type;
  TC_ASSERT(expr->members.size() % group_size == 0);

  for (int i = 0; i < (int)vectorized_children.size(); i++) {
    // TC_P(i);
    auto ch = Expr::create(vectorized_children[i][0]->type);
    ch->members = vectorized_children[i];
    expr->ch.push_back(ch);
  }

  if (expr->type == NodeType::addr) {
    auto addr = expr->members[0]->get_address_();  // TODO:
    if (addr.coeff_aosoa_group_size == 0 || addr.coeff_aosoa_stride == 0) {
      addr.coeff_aosoa_group_size = 0;
      addr.coeff_aosoa_stride = 0;
    }
    expr->get_address_() = addr;
  }
}
}
