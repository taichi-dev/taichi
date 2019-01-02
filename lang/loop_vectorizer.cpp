#include "loop_vectorizer.h"

TLANG_NAMESPACE_BEGIN

void LoopVectorizer::run(Kernel &ker) {
  ker.ret = vectorize(ker.ret);  // vectorize
  ker.stride *= factor;
}

Expr LoopVectorizer::vectorize(Expr node) {
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
        new_node->index_offset(new_j) = node->index_offset(j) + i;
      }
    }
  }

  for (int i = 0; i < node->ch.size(); i++) {
    new_node->ch.push_back(vectorize(node->ch[i]));
  }

  new_node->is_vectorized = true;

  input_to_vectorized[node] = new_node;

  return new_node;
}

TLANG_NAMESPACE_END
