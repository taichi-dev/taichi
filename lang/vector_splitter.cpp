#include "vector_splitter.h"

TLANG_NAMESPACE_BEGIN

Expr VectorSplitter::run(Kernel &kernel, Expr &expr, int target_lanes) {
  if (expr->lanes == target_lanes) {
    return expr;
  }
  this->kernel = &kernel;
  this->target_lanes = target_lanes;
  TC_ASSERT(expr->lanes % target_lanes == 0);
  this->num_splits = expr->lanes / target_lanes;
  TC_INFO("Splitting {} into {}x{}", expr->lanes, num_splits, target_lanes);
  expr.accept(*this);

  auto combined = Expr::create(NodeType::combine);
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
    splits[i] = Expr::create(NodeType::undefined);
    expr->copy_to(*splits[i].ptr());
    for (int j = 0; j < Node::num_additional_values; j++) {
      for (int lane = 0; lane < target_lanes; lane++) {
        splits[i]->attribute(j, lane) =
            splits[i]->attribute(j, target_lanes * i + lane);
      }
    }
    splits[i]->set_lanes(target_lanes);
    for (int j = 0; j < (int)expr->ch.size(); j++) {
      splits[i]->ch[j].set(split[expr->ch[j]][i]);
    }
  }

  if (expr->type == NodeType::adapter_store) {
    for (int i = 0; i < num_splits; i++) {
      for (int j = 0; j < target_lanes; j++) {
        auto &v = splits[i][2]->value<int>(j);
        v = v * num_splits + i;
      }
    }
  }


  split[expr] = splits;
}

TLANG_NAMESPACE_END
