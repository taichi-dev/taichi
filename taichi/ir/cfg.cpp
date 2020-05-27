#include "taichi/ir/cfg.h"
#include <queue>

TLANG_NAMESPACE_BEGIN

CFGNode::CFGNode(Block *block,
                 int begin_location,
                 int end_location,
                 CFGNode *prev_node_in_same_block)
    : block(block),
      begin_location(begin_location),
      end_location(end_location),
      next_node_in_same_block(nullptr) {
  if (empty())
    return;
  TI_ASSERT(begin_location >= 0);
  if (prev_node_in_same_block != nullptr)
    prev_node_in_same_block->next_node_in_same_block = this;
}

void CFGNode::add_edge(CFGNode *from, CFGNode *to) {
  from->next.push_back(to);
  to->prev.push_back(from);
}

bool CFGNode::empty() const {
  return begin_location > end_location;
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

void CFGNode::erase_entire_node() {
  if (empty())
    return;
  int node_size = end_location - begin_location + 1;
  for (int location = end_location; location >= begin_location; location--) {
    block->erase(location);
  }
  end_location -= node_size;  // become empty
  for (auto node = next_node_in_same_block; node != nullptr;
       node = node->next_node_in_same_block) {
    node->begin_location -= node_size;
    node->end_location -= node_size;
  }
}

std::size_t ControlFlowGraph::size() const {
  return nodes.size();
}

CFGNode *ControlFlowGraph::back() {
  return nodes.back().get();
}

bool ControlFlowGraph::unreachable_code_elimination() {
  std::unordered_set<CFGNode *> visited;
  std::queue<CFGNode *> to_visit;
  to_visit.push(nodes[start_node].get());
  visited.insert(nodes[start_node].get());
  // Breadth-first search
  while (!to_visit.empty()) {
    auto now = to_visit.front();
    to_visit.pop();
    for (auto &next : now->next) {
      if (visited.find(next) == visited.end()) {
        to_visit.push(next);
        visited.insert(next);
      }
    }
  }
  bool modified = false;
  for (auto &node : nodes) {
    if (visited.find(node.get()) == visited.end()) {
      // unreachable
      modified = true;
      node->erase_entire_node();
    }
  }
  return modified;
}

TLANG_NAMESPACE_END
