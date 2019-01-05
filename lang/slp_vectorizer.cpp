#include "slp_vectorizer.h"

TLANG_NAMESPACE_BEGIN

// sort all the stores
bool prior_to(Expr &a, Expr &b) {
  TC_ASSERT(a->lanes == 1 && b->lanes == 1);
  TC_ASSERT(a->type == NodeType::pointer && b->type == NodeType::pointer);
  auto sa = a._address()->new_addresses(0), sb = b._address()->new_addresses(0);
  if (sa->parent && sb->parent) {
    // TC_P(sa->parent->child_id(sa));
    // TC_P(sb->parent->child_id(sb));
    return sa->parent->child_id(sa) + 1 == sb->parent->child_id(sb);
  } else {
    return false;
  }
}

void SLPVectorizer::sort(Expr &expr) {
  auto ch = expr->ch;  // a bunch of store nodes
  std::vector<Expr> sorted;

  for (auto c : ch) {
    TC_ASSERT(c->type == NodeType::store);
  }

  while (!ch.empty()) {
    std::vector<Expr> group;
    group.push_back(ch[0]);
    ch.erase(ch.begin());
    while (true) {  // grow
      bool found = false;
      for (int i = 0; i < (int)ch.size(); i++) {  // search
        if (prior_to(ch[i]._pointer(), group.front()._pointer())) {
          group.insert(group.begin(), ch[i]);
          ch.erase(ch.begin() + i);
          found = true;
          break;
        }
        if (prior_to(group.back()._pointer(), ch[i]._pointer())) {
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

Expr SLPVectorizer::run(Expr &expr, int group_size) {
  if (group_size == 1) {
    return expr;
  }
  TC_ASSERT(expr);
  this->group_size = group_size;

  scalar_to_vector.clear();
  // expr should be a ret Op, with its children store Ops.
  // The stores are repeated by a factor of 'pack_size'
  // TC_P(group_size);
  TC_ASSERT(expr->ch.size() % group_size == 0);
  TC_ASSERT(expr->type == NodeType::combine);

  // Create the new root group
  auto combined = Expr::create(NodeType::combine);

  TC_ASSERT(expr->ch.size());

  if (expr->ch[0]->type == NodeType::adapter_store) {
    // cache store
    // for each batch (group)
    for (int k = 0; k < (int)expr->ch.size() / group_size; k++) {
      auto root = Expr::create(NodeType::adapter_store);
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
      for (int i = 0; i < group_size; i++) {
        auto ch = expr->ch[k * group_size + i];
        TC_ASSERT(ch->type == NodeType::store);
        root->members.push_back(ch);  // put scalar inst into vector members
        TC_ASSERT(i < (int)root->members.size());
      }
      root.accept(*this);
      combined->ch.push_back(root);
    }
  } else {
    TC_NOT_IMPLEMENTED
  }
  // TC_P(combined->ch.size());
  combined->lanes = group_size;
  return combined;
}

void SLPVectorizer::visit(Expr &expr) {
  // TC_INFO("Visiting {} {}", expr->id, expr->node_type_name());
  // TC_P(expr->node_type_name());
  // Note: expr may be replaced by an existing vectorized Expr
  if (scalar_to_vector.find(expr->members[0]) != scalar_to_vector.end()) {
    auto existing = scalar_to_vector[expr->members[0]];
    TC_ASSERT(existing->members.size() == expr->members.size());
    bool mismatch = false;
    for (int i = 0; i < (int)existing->members.size(); i++) {
      if (existing->members[i] != expr->members[i]) {
        TC_P(existing->members[i]->id);
        TC_P(expr->members[i]->id);
        TC_P(existing.ptr());
        TC_P(expr.ptr());
        mismatch = true;
      }
    }
    if (mismatch) {
      for (int i = 0; i < (int)existing->members.size(); i++) {
        TC_P(i);
        TC_P(existing->members[i]->id);
        TC_P(existing->members[i]->node_type_name());
        TC_P(expr->members[i]->id);
        TC_P(expr->members[i]->node_type_name());
        TC_WARN_UNLESS(existing->members[i] == expr->members[i], "mismatch");
      }
    }
    TC_ASSERT(!mismatch);
    expr.set(existing);
    // TC_WARN("Using existing {} for {}", existing->id, expr->id);
    return;
  }

  bool first = true;
  std::vector<std::vector<Expr>> vectorized_children;

  expr->set_lanes(group_size);
  auto &m = expr->members;
  for (int i = 0; i < (int)expr->members.size(); i++) {
    expr->data_type = m[0]->data_type;
    expr->binary_type = m[0]->binary_type;
    TC_ASSERT(m[0]->data_type == m[i]->data_type)
    TC_ASSERT(m[0]->binary_type == m[i]->binary_type)  // TODO: fmaddsub
    TC_ASSERT(m[i]->lanes == 1);
    for (int j = 0; j < Node::num_additional_values; j++) {
      expr->attribute(j, i) = m[i]->attribute(j, 0);
      if (expr->type == NodeType::addr) {
        // TC_P(i);
        // TC_P(j);
        // TC_P(expr->attribute<void *>(j, i));
      }
    }
  }

  // Check for isomorphism
  for (auto member : expr->members) {
    // It must not appear to an existing vectorized expr
    // TC_ASSERT(scalar_to_vector.find(member) == scalar_to_vector.end());
    if (first) {
      first = false;
      vectorized_children.resize(member->ch.size());
    } else {
      TC_ASSERT(vectorized_children.size() == member->ch.size());
    }
    for (int i = 0; i < (int)member->ch.size(); i++) {
      vectorized_children[i].push_back(member->ch[i]);
    }
    scalar_to_vector[member] = expr;
  }

  TC_ASSERT(expr->members.size() % group_size == 0);
  for (int i = 0; i < (int)vectorized_children.size(); i++) {
    auto ch = Expr::create(vectorized_children[i][0]->type);
    ch->members = vectorized_children[i];
    expr->ch.push_back(ch);
  }
}

TLANG_NAMESPACE_END
