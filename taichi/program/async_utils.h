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

  bool operator==(const AsyncState &other) const {
    return snode == other.snode && type == other.type;
  }
};

class AsyncStateHash {
 public:
  size_t operator()(const AsyncState &s) const {
    return (uint64)s.snode ^ (uint64)s.type;
  }
};

struct TaskMeta {
  std::string kernel_name;
  SNode *loop_snode{nullptr}; // struct-for only
  std::vector<AsyncState> input_states;
  std::vector<AsyncState> output_states;
};

TLANG_NAMESPACE_END
