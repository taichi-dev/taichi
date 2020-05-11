#pragma once

#include "taichi/ir/ir.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// A basic block in control-flow graph
class CFGNode {
 public:
  Block *block;
  int begin_location, end_location;
  // For updating begin/end locations when modifying the block.
  CFGNode *next_node_in_same_block;

  std::vector<CFGNode *> prev, next;

  CFGNode(Block *block,
          int begin_location,
          int end_location,
          CFGNode *prev_node_in_same_block = nullptr);

  void erase(int location);
  static void add_edge(CFGNode *from, CFGNode *to);
};

class ControlFlowGraph {
 public:
  std::vector<std::unique_ptr<CFGNode>> nodes;
  const int start_node = 0;

  std::size_t size() const {
    return nodes.size();
  }

  template <typename... Args>
  CFGNode *push_back(Args &&... args) {
    nodes.emplace_back(std::make_unique<CFGNode>(std::forward<Args>(args)...));
    return nodes.back().get();
  }
};

TLANG_NAMESPACE_END
