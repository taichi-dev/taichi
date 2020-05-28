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
      prev_node_in_same_block(prev_node_in_same_block),
      next_node_in_same_block(nullptr) {
  if (prev_node_in_same_block != nullptr)
    prev_node_in_same_block->next_node_in_same_block = this;
  if (!empty()) {
    TI_ASSERT(begin_location >= 0);
  }
}

void CFGNode::add_edge(CFGNode *from, CFGNode *to) {
  from->next.push_back(to);
  to->prev.push_back(from);
}

bool CFGNode::empty() const {
  return begin_location > end_location;
}

void CFGNode::erase(int location) {
  TI_ASSERT(location >= begin_location && location <= end_location);
  block->erase(location);
  end_location--;
  for (auto node = next_node_in_same_block; node != nullptr;
       node = node->next_node_in_same_block) {
    node->begin_location--;
    node->end_location--;
  }
}

bool CFGNode::erase_entire_node() {
  if (empty())
    return false;
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
  return true;
}

std::size_t ControlFlowGraph::size() const {
  return nodes.size();
}

CFGNode *ControlFlowGraph::back() {
  return nodes.back().get();
}

void ControlFlowGraph::erase(int position) {
  // Erase an empty node.
  TI_ASSERT(position >= 0 && position < (int)size());
  TI_ASSERT(nodes[position] && nodes[position]->empty());
  if (nodes[position]->prev_node_in_same_block) {
    nodes[position]->prev_node_in_same_block->next_node_in_same_block =
        nodes[position]->next_node_in_same_block;
  }
  if (nodes[position]->next_node_in_same_block) {
    nodes[position]->next_node_in_same_block->prev_node_in_same_block =
        nodes[position]->prev_node_in_same_block;
  }
  for (auto &prev_node : nodes[position]->prev) {
    prev_node->next.erase(std::find(prev_node->next.begin(),
                                    prev_node->next.end(),
                                    nodes[position].get()));
  }
  for (auto &next_node : nodes[position]->next) {
    next_node->prev.erase(std::find(next_node->prev.begin(),
                                    next_node->prev.end(),
                                    nodes[position].get()));
  }
  for (auto &prev_node : nodes[position]->prev) {
    for (auto &next_node : nodes[position]->next) {
      CFGNode::add_edge(prev_node, next_node);
    }
  }
  nodes[position].reset();
}

void ControlFlowGraph::simplify_graph() {
  // Simplify the graph structure, do not modify the IR.
  const int num_nodes = size();
  for (int i = 0; i < num_nodes; i++) {
    if (nodes[i] && nodes[i]->empty() && i != start_node &&
        (nodes[i]->prev.size() <= 1 || nodes[i]->next.size() <= 1)) {
      erase(i);
    }
  }
  int new_num_nodes = 0;
  for (int i = 0; i < num_nodes; i++) {
    if (nodes[i]) {
      if (i != new_num_nodes) {
        nodes[new_num_nodes] = std::move(nodes[i]);
      }
      new_num_nodes++;
    }
  }
  nodes.resize(new_num_nodes);
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
      if (node->erase_entire_node())
        modified = true;
    }
  }
  return modified;
}

TLANG_NAMESPACE_END
