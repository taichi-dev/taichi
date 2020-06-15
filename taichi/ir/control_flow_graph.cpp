#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/analysis.h"
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
  return begin_location >= end_location;
}

void CFGNode::erase(int location) {
  TI_ASSERT(location >= begin_location && location < end_location);
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
  int node_size = end_location - begin_location;
  for (int location = end_location - 1; location >= begin_location;
       location--) {
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

void CFGNode::reaching_definition_analysis() {
  // Calculate reach_gen and reach_kill.
  reach_gen.clear();
  reach_kill.clear();
  for (int i = end_location - 1; i >= begin_location; i--) {
    auto stmt = block->statements[i].get();
    if (auto local_store = stmt->cast<LocalStoreStmt>()) {
      if (!reach_kill_variable(local_store->ptr)) {
        reach_gen.insert(local_store);
        reach_kill.insert(local_store->ptr);
      }
    } else if (auto global_store = stmt->cast<GlobalStoreStmt>()) {
      if (!reach_kill_variable(global_store->ptr)) {
        reach_gen.insert(global_store);
        reach_kill.insert(global_store->ptr);
      }
    } else if (auto atomic = stmt->cast<AtomicOpStmt>()) {
      // Note that we can't do store-to-load forwarding from an AtomicOpStmt.
      if (!reach_kill_variable(atomic->dest)) {
        reach_gen.insert(atomic);
        reach_kill.insert(atomic->dest);
      }
    }
  }
}

bool CFGNode::reach_kill_variable(Stmt *var) const {
  // Does this node (definitely) kill a definition of var?
  return reach_kill.find(var) != reach_kill.end();
  /*if (var->is<AllocaStmt>()) {
    return reach_kill.find(var) != reach_kill.end();
  } else {
    // TODO: How to optimize this?
    for (auto killed_var : reach_kill) {
      if (irpass::analysis::same_statements(var, killed_var)) {
        return true;
      }
    }
    return false;
  }*/
}

Stmt *CFGNode::get_store_forwarding_data(Stmt *var, int position) const {
  // Return the stored data if all definitions in the UD-chain of var at
  // this position store the same data.
  int last_def_position = -1;
  for (auto stmt : reach_gen) {
    if (stmt->store_ptr() == var) {
      int stmt_position = stmt->parent->locate(stmt);
      if (stmt_position < position && stmt_position > last_def_position) {
        last_def_position = stmt_position;
      }
    }
  }
  if (last_def_position != -1) {
    // The UD-chain is inside this node.
    Stmt *result = block->statements[last_def_position]->store_data();
    if (!var->is<AllocaStmt>()) {
      for (int i = last_def_position + 1; i < position; i++) {
        if (maybe_same_address(var, block->statements[i]->store_ptr()) &&
            !irpass::analysis::same_statements(
                result, block->statements[i]->store_data())) {
          return nullptr;
        }
      }
    }
    return result;
  }
  Stmt *result = nullptr;
  auto update_result = [&](Stmt *stmt) {
    if (!result) {
      result = stmt->store_data();
      if (!result) {   // AtomicOpStmt
        return false;  // return nullptr
      }
    } else if (!irpass::analysis::same_statements(result, stmt->store_data())) {
      return false;  // return nullptr
    }
    return true;  // continue
  };
  for (auto stmt : reach_in) {
    if (maybe_same_address(var, stmt->store_ptr())) {
      if (!update_result(stmt))
        return nullptr;
    }
  }
  for (auto stmt : reach_gen) {
    if (maybe_same_address(var, stmt->store_ptr()) &&
        stmt->parent->locate(stmt) < position) {
      if (!update_result(stmt))
        return nullptr;
    }
  }
  if (!result) {
    // The UD-chain is empty.
    // TODO: insert zero if val is alloca
  }
  return result;
}

bool CFGNode::store_to_load_forwarding() {
  bool modified = false;
  for (int i = begin_location; i < end_location; i++) {
    auto stmt = block->statements[i].get();
    Stmt *result = nullptr;
    if (auto local_load = stmt->cast<LocalLoadStmt>()) {
      bool regular = true;
      auto alloca = local_load->ptr[0].var;
      for (int l = 0; l < stmt->width(); l++) {
        if (local_load->ptr[l].offset != l ||
            local_load->ptr[l].var != alloca) {
          regular = false;
        }
      }
      result = get_store_forwarding_data(alloca, i);
    } else if (auto global_load = stmt->cast<GlobalLoadStmt>()) {
      result = get_store_forwarding_data(global_load->ptr, i);
    }
    if (result) {
      stmt->replace_with(result);
      erase(i);  // This causes end_location--
      i--;       // to cancel i++ in the for loop
      modified = true;
    }
  }
  return modified;
}

void ControlFlowGraph::erase(int node_id) {
  // Erase an empty node.
  TI_ASSERT(node_id >= 0 && node_id < (int)size());
  TI_ASSERT(nodes[node_id] && nodes[node_id]->empty());
  if (nodes[node_id]->prev_node_in_same_block) {
    nodes[node_id]->prev_node_in_same_block->next_node_in_same_block =
        nodes[node_id]->next_node_in_same_block;
  }
  if (nodes[node_id]->next_node_in_same_block) {
    nodes[node_id]->next_node_in_same_block->prev_node_in_same_block =
        nodes[node_id]->prev_node_in_same_block;
  }
  for (auto &prev_node : nodes[node_id]->prev) {
    prev_node->next.erase(std::find(
        prev_node->next.begin(), prev_node->next.end(), nodes[node_id].get()));
  }
  for (auto &next_node : nodes[node_id]->next) {
    next_node->prev.erase(std::find(
        next_node->prev.begin(), next_node->prev.end(), nodes[node_id].get()));
  }
  for (auto &prev_node : nodes[node_id]->prev) {
    for (auto &next_node : nodes[node_id]->next) {
      CFGNode::add_edge(prev_node, next_node);
    }
  }
  nodes[node_id].reset();
}

std::size_t ControlFlowGraph::size() const {
  return nodes.size();
}

CFGNode *ControlFlowGraph::back() {
  return nodes.back().get();
}

void ControlFlowGraph::reaching_definition_analysis() {
  const int num_nodes = size();
  std::queue<CFGNode *> to_visit;
  std::unordered_map<CFGNode *, bool> in_queue;
  for (int i = 0; i < num_nodes; i++) {
    nodes[i]->reaching_definition_analysis();
    nodes[i]->reach_in.clear();
    nodes[i]->reach_out = nodes[i]->reach_gen;
    to_visit.push(nodes[i].get());
    in_queue[nodes[i].get()] = true;
  }
  while (!to_visit.empty()) {
    auto now = to_visit.front();
    to_visit.pop();
    in_queue[now] = false;

    now->reach_in.clear();
    for (auto prev_node : now->prev) {
      now->reach_in.insert(prev_node->reach_out.begin(),
                           prev_node->reach_out.end());
    }
    auto old_out = std::move(now->reach_out);
    now->reach_out = now->reach_gen;
    for (auto stmt : now->reach_in) {
      bool killed;
      if (auto local_store = stmt->cast<LocalStoreStmt>()) {
        killed = now->reach_kill_variable(local_store->ptr);
      } else if (auto global_store = stmt->cast<GlobalStoreStmt>()) {
        killed = now->reach_kill_variable(global_store->ptr);
      } else if (auto atomic = stmt->cast<AtomicOpStmt>()) {
        killed = now->reach_kill_variable(atomic->dest);
      } else {
        TI_NOT_IMPLEMENTED
      }
      if (!killed) {
        now->reach_out.insert(stmt);
      }
    }
    if (!(now->reach_out == old_out)) {
      // changed
      for (auto next_node : now->next) {
        if (!in_queue[next_node]) {
          to_visit.push(next_node);
          in_queue[next_node] = true;
        }
      }
    }
  }
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

bool ControlFlowGraph::store_to_load_forwarding() {
  reaching_definition_analysis();
  const int num_nodes = size();
  bool modified = false;
  for (int i = 0; i < num_nodes; i++) {
    if (nodes[i]->store_to_load_forwarding())
      modified = true;
  }
  return modified;
}

TLANG_NAMESPACE_END
