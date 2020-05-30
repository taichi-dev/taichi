#pragma once

#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

// A basic block in control-flow graph
class CFGNode {
 public:
  // This node corresponds to block->statements[i]
  // for i in [begin_location, end_location).
  Block *block;
  int begin_location, end_location;
  // For updating begin/end locations when modifying the block.
  CFGNode *prev_node_in_same_block;
  CFGNode *next_node_in_same_block;

  std::vector<CFGNode *> prev, next;

  CFGNode(Block *block,
          int begin_location,
          int end_location,
          CFGNode *prev_node_in_same_block = nullptr);

  static void add_edge(CFGNode *from, CFGNode *to);
  bool empty() const;
  void erase(int location);
  bool erase_entire_node();
};

class ControlFlowGraph {
 private:
  // Erase an empty node.
  void erase(int node_id);

 public:
  std::vector<std::unique_ptr<CFGNode>> nodes;
  const int start_node = 0;

  template <typename... Args>
  CFGNode *push_back(Args &&... args) {
    nodes.emplace_back(std::make_unique<CFGNode>(std::forward<Args>(args)...));
    return nodes.back().get();
  }

  std::size_t size() const;
  CFGNode *back();

  void simplify_graph();
  bool unreachable_code_elimination();
};

TLANG_NAMESPACE_END
