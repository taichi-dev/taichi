#include "taichi/ir/cfg.h"

TLANG_NAMESPACE_BEGIN

CFGNode::CFGNode(Block *block, int begin_location, int end_location,
                 CFGNode *prev_node_in_same_block)
    : block(block),
      begin_location(begin_location),
      end_location(end_location),
      next_node_in_same_block(nullptr) {
  TI_ASSERT(begin_location >= 0);
  if (prev_node_in_same_block != nullptr)
    prev_node_in_same_block->next_node_in_same_block = this;
}

void CFGNode::erase(int location) {
  block->erase(location);
  end_location--;
  for (auto node = next_node_in_same_block; node != nullptr;
       node = node->next_node_in_same_block) {
    node->begin_location--;
    node->end_location--;
  }
}

void CFGNode::add_edge(CFGNode *from, CFGNode *to) {
  from->next.push_back(to);
  to->prev.push_back(from);
}

TLANG_NAMESPACE_END
