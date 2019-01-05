#include "loop_vectorizer.h"

TLANG_NAMESPACE_BEGIN

void LoopVectorizer::run(Kernel &ker, int factor) {
  this->factor = factor;
  TC_P(factor);
  // simply pick the last index to vectorize
  bool active[max_num_indices];
  std::memset(active, 0, sizeof(active));
  auto s = ker.program.current_snode;
  while (s != nullptr) {
    for (int i = 0; i < max_num_indices; i++) {
      if (s->extractors[i].num_bits) {
        active[i] = true;
        if (vectorized_id < i) {
          vectorized_id = i;
        }
      }
    }
    s = s->parent;
  }
  ker.ret = vectorize(ker.ret);  // vectorize
  ker.stride *= factor;
}

Expr LoopVectorizer::vectorize(Expr node) {
  TC_ASSERT(vectorized_id != -1);
  // TC_P(node->node_type_name());
  if (input_to_vectorized.find(node) != input_to_vectorized.end()) {
    return input_to_vectorized[node];
  }
  auto new_node = Expr::create(node->type);
  new_node->set_lanes(node->lanes * factor);
  new_node->data_type = node->data_type;
  new_node->binary_type = node->binary_type;

  for (int i = 0; i < factor; i++) {
    for (int j = 0; j < node->lanes; j++) {
      int new_j = i * node->lanes + j;
      for (int k = 0; k < Node::num_additional_values; k++) {
        new_node->attribute<double>(k, new_j) = node->attribute<double>(k, j);
      }
    }
  }

  if (node->type == NodeType::index) {
    for (int i = 0; i < factor; i++) {
      for (int j = 0; j < node->lanes; j++) {
        int new_j = i * node->lanes + j;
        new_node->index_offset(new_j) =
            node->index_offset(j) +
            i * (node->index_id(0) == vectorized_id);
      }
    }
  }

  for (int i = 0; i < node->ch.size(); i++) {
    new_node->ch.push_back(vectorize(node->ch[i]));
  }

  input_to_vectorized[node] = new_node;

  return new_node;
}

TLANG_NAMESPACE_END
