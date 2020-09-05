#pragma once

#include <unordered_set>

#include "taichi/ir/snode.h"

TLANG_NAMESPACE_BEGIN

struct AsyncState {
  enum class Type { mask, value, list };

  AsyncState(SNode *snode, Type type) : snode(snode), type(type) {
  }

  SNode *snode;
  Type type;

  bool operator<(const AsyncState &other) const {
    return snode < other.snode || (snode == other.snode && type < other.type);
  }
};

struct TaskMeta {
  SNode *loop_snode{nullptr};
  std::vector<AsyncState> input_states;
  std::vector<AsyncState> output_states;
};

TLANG_NAMESPACE_END
