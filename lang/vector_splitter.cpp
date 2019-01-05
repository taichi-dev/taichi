#include "vector_splitter.h"

TLANG_NAMESPACE_BEGIN

Expr VectorSplitter::run(Kernel &kernel, Expr &expr, int target_lanes) {
  this->kernel = &kernel;
  this->target_lanes = target_lanes;
  TC_ASSERT(expr->lanes % target_lanes == 0);
  this->num_splits = expr->lanes / target_lanes;
  visit(expr);

  Expr combined;
  combined->lanes = target_lanes;
  for (auto &c : expr->ch) {
    for (auto &v : split[c]) {
      combined->ch.push_back(v);
    }
  }
  return combined;
}

void VectorSplitter::visit(Expr &expr) {
  if (split.find(expr) != split.end()) {
    return;
  }

  // create #num_splits splits

  std::vector<Expr> splits(num_splits);
  for (int i = 0; i < num_splits; i++) {
    expr->copy_to(*splits[i].ptr());
    for (int j = 0; j < Node::num_additional_values; j++) {
      for (int lane = 0; lane < target_lanes; lane++) {
        splits[i]->attribute(j, lane) =
            splits[i]->attribute(j, target_lanes * i + lane);
      }
    }
    splits[i]->set_lanes(target_lanes);
    for (int j = 0; j < expr->ch.size(); j++) {
      splits[i]->ch[j] = split[expr->ch[j]][i];
    }
  }

  // treat adapter_load/store
  if (expr->type == NodeType::adapter_store) {
    for (int j = 0; j < num_splits; j++) {
      for (int i = 0; i < target_lanes; i++) {
        auto &offset = splits[j]->ch[2]->attribute(0, i);
        offset = offset * num_splits + j;
      }
    }
  } else if (expr->type == NodeType::adapter_load) {
    // input_num_splits;
    for (int j = 0; j < num_splits; j++) {
      for (int i = 0; i < target_lanes; i++) {
        auto &offset = splits[j]->ch[2]->attribute(0, i);
        offset = offset * num_splits + j;
      }
    }
  }

  split[expr] = splits;
}

TLANG_NAMESPACE_END
