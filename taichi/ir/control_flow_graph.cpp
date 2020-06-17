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
  auto parent_block = block;
  parent_blocks.insert(parent_block);
  while (parent_block->parent) {
    parent_block = parent_block->parent;
    parent_blocks.insert(parent_block);
  }
}

void CFGNode::add_edge(CFGNode *from, CFGNode *to) {
  from->next.push_back(to);
  to->prev.push_back(from);
}

bool CFGNode::empty() const {
  return begin_location >= end_location;
}

std::size_t CFGNode::size() const {
  return end_location - begin_location;
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

void CFGNode::insert(std::unique_ptr<Stmt> &&new_stmt, int location) {
  TI_ASSERT(location >= begin_location && location <= end_location);
  block->insert(std::move(new_stmt), location);
  end_location++;
  for (auto node = next_node_in_same_block; node != nullptr;
       node = node->next_node_in_same_block) {
    node->begin_location++;
    node->end_location++;
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

void CFGNode::reaching_definition_analysis(bool after_lower_access) {
  // Calculate reach_gen and reach_kill.
  reach_gen.clear();
  reach_kill.clear();
  for (int i = end_location - 1; i >= begin_location; i--) {
    // loop in reversed order
    auto stmt = block->statements[i].get();
    auto data_source_ptr = irpass::analysis::get_store_destination(stmt);
    if (data_source_ptr) {
      // stmt provides a data source
      if (after_lower_access &&
          !(stmt->is<AllocaStmt>() || stmt->is<LocalStoreStmt>())) {
        continue;
      }
      // TODO: If stmt is a GlobalPtrStmt or a GlobalTemporaryStmt, we may lose
      // optimization opportunities in this case (we could replace $4 with b):
      // $1: global ptr a
      // $2: global store [$1 <- b]
      // $3(stmt): global ptr a
      // $4: global load $3
      if (!reach_kill_variable(data_source_ptr)) {
        reach_gen.insert(stmt);
        reach_kill.insert(data_source_ptr);
      }
    }
  }
}

bool CFGNode::reach_kill_variable(Stmt *var) const {
  // Does this node (definitely) kill a definition of var?
  // return reach_kill.find(var) != reach_kill.end();
  if (var->is<AllocaStmt>()) {
    return reach_kill.find(var) != reach_kill.end();
  } else {
    // TODO: How to optimize this?
    for (auto killed_var : reach_kill) {
      if (irpass::analysis::same_statements(var, killed_var)) {
        return true;
      }
    }
    return false;
  }
}

Stmt *CFGNode::get_store_forwarding_data(Stmt *var, int position) const {
  // Return the stored data if all definitions in the UD-chain of var at
  // this position store the same data.
  int last_def_position = -1;
  for (int i = position - 1; i >= begin_location; i--) {
    if (irpass::analysis::get_store_destination(block->statements[i].get()) ==
        var) {
      last_def_position = i;
      break;
    }
  }
  if (last_def_position != -1) {
    // The UD-chain is inside this node.
    Stmt *result = irpass::analysis::get_store_data(
        block->statements[last_def_position].get());
    if (!var->is<AllocaStmt>()) {
      for (int i = last_def_position + 1; i < position; i++) {
        if (maybe_same_address(var, irpass::analysis::get_store_destination(
                                        block->statements[i].get())) &&
            !irpass::analysis::same_statements(
                result,
                irpass::analysis::get_store_data(block->statements[i].get()))) {
          return nullptr;
        }
      }
    }
    return result;
  }
  Stmt *result = nullptr;
  auto visible = [&](Stmt *stmt) {
    // Do we need to check if `stmt` is before `position` here?
    return parent_blocks.find(stmt->parent) != parent_blocks.end();
  };
  auto update_result = [&](Stmt *stmt) {
    auto data = irpass::analysis::get_store_data(stmt);
    if (!data) {     // not forwardable
      return false;  // return nullptr
    }
    if (!result) {
      result = data;
    } else if (!irpass::analysis::same_statements(result, data)) {
      // check the special case of alloca (initialized to 0)
      if (!(result->is<AllocaStmt>() && data->is<ConstStmt>() &&
            data->width() == 1 &&
            data->as<ConstStmt>()->val[0].equal_value(0))) {
        return false;  // return nullptr
      }
    }
    if (visible(data))
      result = data;
    return true;  // continue the following loops
  };
  for (auto stmt : reach_in) {
    if (maybe_same_address(var,
                           irpass::analysis::get_store_destination(stmt))) {
      if (!update_result(stmt))
        return nullptr;
    }
  }
  for (auto stmt : reach_gen) {
    if (maybe_same_address(var,
                           irpass::analysis::get_store_destination(stmt)) &&
        stmt->parent->locate(stmt) < position) {
      if (!update_result(stmt))
        return nullptr;
    }
  }
  if (!result) {
    // The UD-chain is empty.
    TI_WARN("stmt {} loaded in stmt {} before storing.", var->id,
            block->statements[position]->id);
    return nullptr;
  }
  if (!visible(result)) {
    return nullptr;
  }
  return result;
}

bool CFGNode::store_to_load_forwarding(bool after_lower_access) {
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
      if (regular) {
        result = get_store_forwarding_data(alloca, i);
      }
    } else if (auto global_load = stmt->cast<GlobalLoadStmt>()) {
      if (!after_lower_access) {
        result = get_store_forwarding_data(global_load->ptr, i);
      }
    }
    if (result) {
      if (result->is<AllocaStmt>()) {
        // special case of alloca (initialized to 0)
        auto zero = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(0));
        stmt->replace_with(zero.get());
        erase(i);
        insert(std::move(zero), i);
      } else {
        stmt->replace_with(result);
        erase(i);  // This causes end_location--
        i--;       // to cancel i++ in the for loop
        modified = true;
      }
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

void ControlFlowGraph::print_graph_structure() const {
  const int num_nodes = size();
  std::cout << "Control Flow Graph with " << num_nodes
            << " nodes:" << std::endl;
  std::unordered_map<CFGNode *, int> to_index;
  for (int i = 0; i < num_nodes; i++) {
    to_index[nodes[i].get()] = i;
  }
  for (int i = 0; i < num_nodes; i++) {
    std::string node_info = fmt::format("Node {} : ", i);
    if (nodes[i]->empty()) {
      node_info += "empty";
    } else {
      node_info += fmt::format(
          "{}~{} (size={})",
          nodes[i]->block->statements[nodes[i]->begin_location]->name(),
          nodes[i]->block->statements[nodes[i]->end_location - 1]->name(),
          nodes[i]->size());
    }
    if (!nodes[i]->prev.empty()) {
      std::vector<std::string> indices;
      for (auto prev_node : nodes[i]->prev) {
        indices.push_back(std::to_string(to_index[prev_node]));
      }
      node_info += fmt::format("; prev={{{}}}", fmt::join(indices, ", "));
    }
    if (!nodes[i]->next.empty()) {
      std::vector<std::string> indices;
      for (auto next_node : nodes[i]->next) {
        indices.push_back(std::to_string(to_index[next_node]));
      }
      node_info += fmt::format("; next={{{}}}", fmt::join(indices, ", "));
    }
    std::cout << node_info << std::endl;
  }
}

void ControlFlowGraph::reaching_definition_analysis(bool after_lower_access) {
  TI_AUTO_PROF;
  const int num_nodes = size();
  std::queue<CFGNode *> to_visit;
  std::unordered_map<CFGNode *, bool> in_queue;
  for (int i = 0; i < num_nodes; i++) {
    nodes[i]->reaching_definition_analysis(after_lower_access);
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
      if (!now->reach_kill_variable(
              irpass::analysis::get_store_destination(stmt))) {
        now->reach_out.insert(stmt);
      }
    }
    if (now->reach_out != old_out) {
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
  TI_AUTO_PROF;
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

bool ControlFlowGraph::store_to_load_forwarding(bool after_lower_access) {
  TI_AUTO_PROF;
  reaching_definition_analysis(after_lower_access);
  const int num_nodes = size();
  bool modified = false;
  for (int i = 0; i < num_nodes; i++) {
    if (nodes[i]->store_to_load_forwarding(after_lower_access))
      modified = true;
  }
  return modified;
}

TLANG_NAMESPACE_END
