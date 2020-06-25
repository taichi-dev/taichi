#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include <queue>

TLANG_NAMESPACE_BEGIN

CFGNode::CFGNode(Block *block,
                 int begin_location,
                 int end_location,
                 bool is_parallel_executed,
                 CFGNode *prev_node_in_same_block)
    : block(block),
      begin_location(begin_location),
      end_location(end_location),
      is_parallel_executed(is_parallel_executed),
      prev_node_in_same_block(prev_node_in_same_block),
      next_node_in_same_block(nullptr) {
  if (prev_node_in_same_block != nullptr)
    prev_node_in_same_block->next_node_in_same_block = this;
  if (!empty()) {
    TI_ASSERT(begin_location >= 0);
    TI_ASSERT(block);
    auto parent_block = block;
    parent_blocks.insert(parent_block);
    while (parent_block->parent) {
      parent_block = parent_block->parent;
      parent_blocks.insert(parent_block);
    }
  }
}

CFGNode::CFGNode() : CFGNode(nullptr, -1, -1, false, nullptr) {
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

void CFGNode::replace_with(int location,
                           std::unique_ptr<Stmt> &&new_stmt,
                           bool replace_usages) {
  TI_ASSERT(location >= begin_location && location < end_location);
  block->replace_with(block->statements[location].get(), std::move(new_stmt),
                      replace_usages);
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

bool CFGNode::contain_variable(const std::unordered_set<Stmt *> &var_set,
                               Stmt *var) {
  if (var->is<AllocaStmt>() || var->is<StackAllocaStmt>()) {
    return var_set.find(var) != var_set.end();
  } else {
    // TODO: How to optimize this?
    for (auto set_var : var_set) {
      if (irpass::analysis::same_statements(var, set_var)) {
        return true;
      }
    }
    return false;
  }
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
        // After lower_access, we only analyze local variables.
        continue;
      }
      if (!reach_kill_variable(data_source_ptr)) {
        reach_gen.insert(stmt);
        reach_kill.insert(data_source_ptr);
      }
    }
  }
}

bool CFGNode::reach_kill_variable(Stmt *var) const {
  // Does this node (definitely) kill a definition of var?
  return contain_variable(reach_kill, var);
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
    // var == stmt is for the case that a global ptr is never stored.
    // In this case, stmt is from nodes[start_node]->reach_gen.
    if (var == stmt ||
        maybe_same_address(var,
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
        auto zero =
            Stmt::make<ConstStmt>(TypedConstant(result->ret_type.data_type, 0));
        zero->repeat(result->width());
        replace_with(i, std::move(zero), true);
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

void CFGNode::live_variable_analysis(bool after_lower_access) {
  live_gen.clear();
  live_kill.clear();
  for (int i = begin_location; i < end_location; i++) {
    auto stmt = block->statements[i].get();
    auto load_ptrs = irpass::analysis::get_load_pointers(stmt);
    for (auto &load_ptr : load_ptrs) {
      if (!after_lower_access ||
          (load_ptr->is<AllocaStmt>() || load_ptr->is<StackAllocaStmt>())) {
        // After lower_access, we only analyze local variables and stacks.
        if (!contain_variable(live_kill, load_ptr)) {
          live_gen.insert(load_ptr);
        }
      }
    }
    auto store_ptr = irpass::analysis::get_store_destination(stmt);
    // TODO: Consider stacks in get_store_destination instead of here
    //  for store-to-load forwarding on stacks
    if (auto stack_pop = stmt->cast<StackPopStmt>()) {
      store_ptr = stack_pop->stack;
    } else if (auto stack_push = stmt->cast<StackPushStmt>()) {
      store_ptr = stack_push->stack;
    } else if (auto stack_acc_adj = stmt->cast<StackAccAdjointStmt>()) {
      store_ptr = stack_acc_adj->stack;
    }
    if (store_ptr) {
      if (!after_lower_access ||
          (store_ptr->is<AllocaStmt>() || store_ptr->is<StackAllocaStmt>())) {
        // After lower_access, we only analyze local variables and stacks.
        live_kill.insert(store_ptr);
      }
    }
  }
}

bool CFGNode::dead_store_elimination(bool after_lower_access) {
  bool modified = false;
  std::unordered_set<Stmt *> live_in_this_node;
  for (int i = end_location - 1; i >= begin_location; i--) {
    auto stmt = block->statements[i].get();
    auto store_ptr = irpass::analysis::get_store_destination(stmt);
    // TODO: Consider stacks in get_store_destination instead of here
    //  for store-to-load forwarding on stacks
    if (auto stack_pop = stmt->cast<StackPopStmt>()) {
      store_ptr = stack_pop->stack;
    } else if (auto stack_push = stmt->cast<StackPushStmt>()) {
      store_ptr = stack_push->stack;
    } else if (auto stack_acc_adj = stmt->cast<StackAccAdjointStmt>()) {
      store_ptr = stack_acc_adj->stack;
    }
    if (store_ptr) {
      if (!after_lower_access ||
          (store_ptr->is<AllocaStmt>() || store_ptr->is<StackAllocaStmt>())) {
        // After lower_access, we only analyze local variables and stacks.
        // Do not eliminate AllocaStmt here.
        if (!stmt->is<AllocaStmt>() && !contain_variable(live_out, store_ptr) &&
            !contain_variable(live_in_this_node, store_ptr)) {
          // Neither used in other nodes nor used in this node.
          if (auto atomic = stmt->cast<AtomicOpStmt>()) {
            // Weaken the atomic operation to a load.
            if (atomic->dest->is<AllocaStmt>()) {
              auto local_load =
                  Stmt::make<LocalLoadStmt>(LocalAddress(atomic->dest, 0));
              local_load->ret_type = atomic->ret_type;
              replace_with(i, std::move(local_load), true);
              // Notice that we have a load here.
              live_in_this_node.insert(atomic->dest);
              modified = true;
              continue;
            } else if (!is_parallel_executed) {
              // If this node is parallel executed, we can't weaken a global
              // atomic operation to a global load.
              // TODO: analyze if atomic->dest is never accessed by other
              //  threads.
              auto global_load = Stmt::make<GlobalLoadStmt>(atomic->dest);
              global_load->ret_type = atomic->ret_type;
              replace_with(i, std::move(global_load), true);
              // Notice that we have a load here.
              live_in_this_node.insert(atomic->dest);
              modified = true;
              continue;
            }
          } else {
            erase(i);
            modified = true;
            continue;
          }
        } else {
          // A non-eliminated store.
          live_in_this_node.erase(store_ptr);
        }
      }
    }
    auto load_ptrs = irpass::analysis::get_load_pointers(stmt);
    for (auto &load_ptr : load_ptrs) {
      if (!after_lower_access ||
          (load_ptr->is<AllocaStmt>() || load_ptr->is<StackAllocaStmt>())) {
        // After lower_access, we only analyze local variables and stacks.
        live_in_this_node.insert(load_ptr);
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
    if (!nodes[i]->live_in.empty()) {
      std::vector<std::string> indices;
      for (auto stmt : nodes[i]->live_in) {
        indices.push_back(stmt->name());
      }
      node_info += fmt::format("; live_in={{{}}}", fmt::join(indices, ", "));
    }
    if (!nodes[i]->live_out.empty()) {
      std::vector<std::string> indices;
      for (auto stmt : nodes[i]->live_out) {
        indices.push_back(stmt->name());
      }
      node_info += fmt::format("; live_out={{{}}}", fmt::join(indices, ", "));
    }
    std::cout << node_info << std::endl;
  }
}

void ControlFlowGraph::reaching_definition_analysis(bool after_lower_access) {
  TI_AUTO_PROF;
  const int num_nodes = size();
  std::queue<CFGNode *> to_visit;
  std::unordered_map<CFGNode *, bool> in_queue;
  TI_ASSERT(nodes[start_node]->empty());
  nodes[start_node]->reach_gen.clear();
  nodes[start_node]->reach_kill.clear();
  if (!after_lower_access) {
    for (int i = 0; i < num_nodes; i++) {
      for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
        if (auto global_load =
                nodes[i]->block->statements[j]->cast<GlobalLoadStmt>()) {
          nodes[start_node]->reach_gen.insert(global_load->ptr);
        }
        // Since we only do store-to-load forwarding, we don't need to mark the
        // start node as data sources of other global pointers.
      }
    }
  }
  for (int i = 0; i < num_nodes; i++) {
    if (i != start_node) {
      nodes[i]->reaching_definition_analysis(after_lower_access);
    }
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

void ControlFlowGraph::live_variable_analysis(bool after_lower_access) {
  TI_AUTO_PROF;
  const int num_nodes = size();
  std::queue<CFGNode *> to_visit;
  std::unordered_map<CFGNode *, bool> in_queue;
  TI_ASSERT(nodes[end_node]->empty());
  nodes[end_node]->live_gen.clear();
  nodes[end_node]->live_kill.clear();
  if (!after_lower_access) {
    for (int i = 0; i < num_nodes; i++) {
      for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
        auto stmt = nodes[i]->block->statements[j].get();
        auto store_ptr = irpass::analysis::get_store_destination(stmt);
        if (store_ptr && !store_ptr->is<AllocaStmt>() &&
            !store_ptr->is<StackAllocaStmt>()) {
          // A global pointer that may be loaded after this kernel.
          nodes[end_node]->live_gen.insert(store_ptr);
        }
      }
    }
  }
  for (int i = num_nodes - 1; i >= 0; i--) {
    if (i != end_node) {
      nodes[i]->live_variable_analysis(after_lower_access);
    }
    nodes[i]->live_out.clear();
    nodes[i]->live_in = nodes[i]->live_gen;
    to_visit.push(nodes[i].get());
    in_queue[nodes[i].get()] = true;
  }
  while (!to_visit.empty()) {
    auto now = to_visit.front();
    to_visit.pop();
    in_queue[now] = false;

    now->live_out.clear();
    for (auto next_node : now->next) {
      now->live_out.insert(next_node->live_in.begin(),
                           next_node->live_in.end());
    }
    auto old_in = std::move(now->live_in);
    now->live_in = now->live_gen;
    for (auto stmt : now->live_out) {
      if (!CFGNode::contain_variable(now->live_kill, stmt)) {
        now->live_in.insert(stmt);
      }
    }
    if (now->live_in != old_in) {
      // changed
      for (auto prev_node : now->prev) {
        if (!in_queue[prev_node]) {
          to_visit.push(prev_node);
          in_queue[prev_node] = true;
        }
      }
    }
  }
}

void ControlFlowGraph::simplify_graph() {
  // Simplify the graph structure, do not modify the IR.
  const int num_nodes = size();
  while (true) {
    bool modified = false;
    for (int i = 0; i < num_nodes; i++) {
      if (nodes[i] && nodes[i]->empty() && i != start_node && i != end_node &&
          (nodes[i]->prev.size() <= 1 || nodes[i]->next.size() <= 1)) {
        erase(i);
        modified = true;
      }
    }
    if (!modified)
      break;
  }
  int new_num_nodes = 0;
  for (int i = 0; i < num_nodes; i++) {
    if (nodes[i]) {
      if (i != new_num_nodes) {
        nodes[new_num_nodes] = std::move(nodes[i]);
      }
      if (end_node == i) {
        end_node = new_num_nodes;
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

bool ControlFlowGraph::dead_store_elimination(bool after_lower_access) {
  TI_AUTO_PROF;
  live_variable_analysis(after_lower_access);
  const int num_nodes = size();
  bool modified = false;
  for (int i = 0; i < num_nodes; i++) {
    if (nodes[i]->dead_store_elimination(after_lower_access))
      modified = true;
  }
  return modified;
}

TLANG_NAMESPACE_END
