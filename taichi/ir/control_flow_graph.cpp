#include "taichi/ir/control_flow_graph.h"

#include <queue>
#include <unordered_set>

#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/system/profiler.h"

namespace taichi {
namespace lang {

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
    // For non-empty nodes, precompute |parent_blocks| to accelerate
    // get_store_forwarding_data().
    TI_ASSERT(begin_location >= 0);
    TI_ASSERT(block);
    auto parent_block = block;
    parent_blocks_.insert(parent_block);
    while (parent_block->parent_block()) {
      parent_block = parent_block->parent_block();
      parent_blocks_.insert(parent_block);
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
                           bool replace_usages) const {
  TI_ASSERT(location >= begin_location && location < end_location);
  block->replace_with(block->statements[location].get(), std::move(new_stmt),
                      replace_usages);
}

bool CFGNode::contain_variable(const std::unordered_set<Stmt *> &var_set,
                               Stmt *var) {
  if (var->is<AllocaStmt>() || var->is<AdStackAllocaStmt>()) {
    return var_set.find(var) != var_set.end();
  } else {
    // TODO: How to optimize this?
    if (var_set.find(var) != var_set.end())
      return true;
    return std::any_of(var_set.begin(), var_set.end(), [&](Stmt *set_var) {
      return irpass::analysis::definitely_same_address(var, set_var);
    });
  }
}

bool CFGNode::may_contain_variable(const std::unordered_set<Stmt *> &var_set,
                                   Stmt *var) {
  if (var->is<AllocaStmt>() || var->is<AdStackAllocaStmt>()) {
    return var_set.find(var) != var_set.end();
  } else {
    // TODO: How to optimize this?
    if (var_set.find(var) != var_set.end())
      return true;
    return std::any_of(var_set.begin(), var_set.end(), [&](Stmt *set_var) {
      return irpass::analysis::maybe_same_address(var, set_var);
    });
  }
}

bool CFGNode::reach_kill_variable(Stmt *var) const {
  // Does this node (definitely) kill a definition of var?
  return contain_variable(reach_kill, var);
}

Stmt *CFGNode::get_store_forwarding_data(Stmt *var, int position) const {
  // Return the stored data if all definitions in the UD-chain of |var| at
  // this position store the same data.
  int last_def_position = -1;
  for (int i = position - 1; i >= begin_location; i--) {
    if (block->statements[i]->is<FuncCallStmt>()) {
      return nullptr;
    }
    for (auto store_ptr :
         irpass::analysis::get_store_destination(block->statements[i].get())) {
      if (irpass::analysis::definitely_same_address(var, store_ptr)) {
        last_def_position = i;
        break;
      }
    }
    if (last_def_position != -1) {
      break;
    }
  }
  auto may_contain_address = [](Stmt *store_stmt, Stmt *var) {
    for (auto store_ptr : irpass::analysis::get_store_destination(store_stmt)) {
      if (irpass::analysis::maybe_same_address(var, store_ptr)) {
        return true;
      }
    }
    return false;
  };
  if (last_def_position != -1) {
    // The UD-chain is inside this node.
    Stmt *result = irpass::analysis::get_store_data(
        block->statements[last_def_position].get());
    if (!var->is<AllocaStmt>()) {
      for (int i = last_def_position + 1; i < position; i++) {
        if (!irpass::analysis::same_value(
                result,
                irpass::analysis::get_store_data(block->statements[i].get()))) {
          if (may_contain_address(block->statements[i].get(), var)) {
            return nullptr;
          }
        }
      }
    }
    return result;
  }
  Stmt *result = nullptr;
  bool result_visible = false;
  auto visible = [&](Stmt *stmt) {
    // Check if |stmt| is before |position| here.
    if (stmt->parent == block) {
      return stmt->parent->locate(stmt) < position;
    }
    // |parent_blocks| is precomputed in the constructor of CFGNode.
    // TODO: What if |stmt| appears in an ancestor of |block| but after
    //  |position|?
    return parent_blocks_.find(stmt->parent) != parent_blocks_.end();
  };
  /**
   * |stmt| is a definition in the UD-chain of |var|. Update |result| with
   * |stmt|. If either the stored data of |stmt| is not forwardable or the
   * stored data of |stmt| is not definitely the same as other definitions of
   * |var|, return false to show that there is no store-to-load forwardable
   * data.
   */
  auto update_result = [&](Stmt *stmt) {
    auto data = irpass::analysis::get_store_data(stmt);
    if (!data) {     // not forwardable
      return false;  // return nullptr
    }
    if (!result) {
      result = data;
      result_visible = visible(data);
      return true;  // continue the following loops
    }
    if (!irpass::analysis::same_value(result, data)) {
      // check the special case of alloca (initialized to 0)
      if (!(result->is<AllocaStmt>() && data->is<ConstStmt>() &&
            data->as<ConstStmt>()->val.equal_value(0))) {
        return false;  // return nullptr
      }
    }
    if (!result_visible && visible(data)) {
      // pick the visible one for store-to-load forwarding
      result = data;
      result_visible = true;
    }
    return true;  // continue the following loops
  };
  for (auto stmt : reach_in) {
    // var == stmt is for the case that a global ptr is never stored.
    // In this case, stmt is from nodes[start_node]->reach_gen.
    if (var == stmt || may_contain_address(stmt, var)) {
      if (!update_result(stmt))
        return nullptr;
    }
  }
  for (auto stmt : reach_gen) {
    if (may_contain_address(stmt, var) &&
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
  if (!result_visible) {
    // The data is store-to-load forwardable but not visible at the place we
    // are going to forward. We cannot forward it in this case.
    return nullptr;
  }
  return result;
}

void CFGNode::reaching_definition_analysis(bool after_lower_access) {
  // Calculate |reach_gen| and |reach_kill|.
  reach_gen.clear();
  reach_kill.clear();
  for (int i = end_location - 1; i >= begin_location; i--) {
    // loop in reversed order
    auto stmt = block->statements[i].get();
    auto data_source_ptrs = irpass::analysis::get_store_destination(stmt);
    for (auto data_source_ptr : data_source_ptrs) {
      // stmt provides a data source
      if (after_lower_access && !(data_source_ptr->is<AllocaStmt>())) {
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

bool CFGNode::store_to_load_forwarding(bool after_lower_access,
                                       bool autodiff_enabled) {
  bool modified = false;
  for (int i = begin_location; i < end_location; i++) {
    // Store-to-load forwarding
    auto stmt = block->statements[i].get();
    Stmt *result = nullptr;
    if (auto local_load = stmt->cast<LocalLoadStmt>()) {
      result = get_store_forwarding_data(local_load->src, i);
    } else if (auto global_load = stmt->cast<GlobalLoadStmt>()) {
      if (!after_lower_access && !autodiff_enabled) {
        result = get_store_forwarding_data(global_load->src, i);
      }
    }
    if (result) {
      // Forward the stored data |result|.
      if (result->is<AllocaStmt>()) {
        // special case of alloca (initialized to 0)
        auto zero = Stmt::make<ConstStmt>(TypedConstant(result->ret_type, 0));
        replace_with(i, std::move(zero), true);
      } else {
        stmt->replace_usages_with(result);
        erase(i);  // This causes end_location--
        i--;       // to cancel i++ in the for loop
        modified = true;
      }
      continue;
    }

    // Identical store elimination
    if (auto local_store = stmt->cast<LocalStoreStmt>()) {
      result = get_store_forwarding_data(local_store->dest, i);
      if (result && result->is<AllocaStmt>() && !autodiff_enabled) {
        // special case of alloca (initialized to 0)
        if (auto stored_data = local_store->val->cast<ConstStmt>()) {
          if (stored_data->val.equal_value(0)) {
            erase(i);  // This causes end_location--
            i--;       // to cancel i++ in the for loop
            modified = true;
          }
        }
      } else {
        // not alloca
        if (irpass::analysis::same_value(result, local_store->val)) {
          erase(i);  // This causes end_location--
          i--;       // to cancel i++ in the for loop
          modified = true;
        }
      }
    } else if (auto global_store = stmt->cast<GlobalStoreStmt>()) {
      if (!after_lower_access) {
        result = get_store_forwarding_data(global_store->dest, i);
        if (irpass::analysis::same_value(result, global_store->val)) {
          erase(i);  // This causes end_location--
          i--;       // to cancel i++ in the for loop
          modified = true;
        }
      }
    }
  }
  return modified;
}

void CFGNode::gather_loaded_snodes(std::unordered_set<SNode *> &snodes) const {
  // Gather the SNodes which this CFGNode loads.
  // Requires reaching definition analysis.
  std::unordered_set<Stmt *> killed_in_this_node;
  for (int i = begin_location; i < end_location; i++) {
    auto stmt = block->statements[i].get();
    auto load_ptrs = irpass::analysis::get_load_pointers(stmt);
    for (auto &load_ptr : load_ptrs) {
      if (auto global_ptr = load_ptr->cast<GlobalPtrStmt>()) {
        // Avoid computing the UD-chain if every SNode in this global ptr
        // are already loaded because it can be time-consuming.
        auto snode = global_ptr->snode;
        if (snodes.count(snode) > 0) {
          continue;
        }
        if (reach_in.find(global_ptr) != reach_in.end() &&
            !contain_variable(killed_in_this_node, global_ptr)) {
          // The UD-chain contains the value before this offloaded task.
          snodes.insert(snode);
        }
      }
    }
    auto store_ptrs = irpass::analysis::get_store_destination(stmt);
    for (auto &store_ptr : store_ptrs) {
      if (store_ptr->is<GlobalPtrStmt>()) {
        killed_in_this_node.insert(store_ptr);
      }
    }
  }
}

void CFGNode::live_variable_analysis(bool after_lower_access) {
  live_gen.clear();
  live_kill.clear();
  for (int i = begin_location; i < end_location; i++) {
    auto stmt = block->statements[i].get();
    auto load_ptrs = irpass::analysis::get_load_pointers(stmt);
    for (auto &load_ptr : load_ptrs) {
      if (!after_lower_access ||
          (load_ptr->is<AllocaStmt>() || load_ptr->is<AdStackAllocaStmt>())) {
        // After lower_access, we only analyze local variables and stacks.
        if (!contain_variable(live_kill, load_ptr)) {
          live_gen.insert(load_ptr);
        }
      }
    }
    auto store_ptrs = irpass::analysis::get_store_destination(stmt);
    // TODO: Consider AD-stacks in get_store_destination instead of here
    //  for store-to-load forwarding on AD-stacks
    // TODO: SNode deactivation is also a definite store
    if (auto stack_pop = stmt->cast<AdStackPopStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_pop->stack);
    } else if (auto stack_push = stmt->cast<AdStackPushStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_push->stack);
    } else if (auto stack_acc_adj = stmt->cast<AdStackAccAdjointStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_acc_adj->stack);
    }
    for (auto store_ptr : store_ptrs) {
      if (!after_lower_access ||
          (store_ptr->is<AllocaStmt>() || store_ptr->is<AdStackAllocaStmt>())) {
        // After lower_access, we only analyze local variables and stacks.
        live_kill.insert(store_ptr);
      }
    }
  }
}

bool CFGNode::dead_store_elimination(bool after_lower_access) {
  bool modified = false;
  std::unordered_set<Stmt *> live_in_this_node;
  std::unordered_set<Stmt *> killed_in_this_node;
  // Map a variable to its nearest load
  std::unordered_map<Stmt *, Stmt *> live_load_in_this_node;
  for (int i = end_location - 1; i >= begin_location; i--) {
    auto stmt = block->statements[i].get();
    if (stmt->is<FuncCallStmt>()) {
      killed_in_this_node.clear();
      live_load_in_this_node.clear();
    }
    auto store_ptrs = irpass::analysis::get_store_destination(stmt);
    // TODO: Consider AD-stacks in get_store_destination instead of here
    //  for store-to-load forwarding on AD-stacks
    if (auto stack_pop = stmt->cast<AdStackPopStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_pop->stack);
    } else if (auto stack_push = stmt->cast<AdStackPushStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_push->stack);
    } else if (auto stack_acc_adj = stmt->cast<AdStackAccAdjointStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_acc_adj->stack);
    } else if (stmt->is<AdStackAllocaStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stmt);
    }
    if (store_ptrs.size() == 1) {
      // Dead store elimination
      auto store_ptr = store_ptrs.front();
      if (!after_lower_access ||
          (store_ptr->is<AllocaStmt>() || store_ptr->is<AdStackAllocaStmt>())) {
        // After lower_access, we only analyze local variables and stacks.
        // Do not eliminate AllocaStmt and AdStackAllocaStmt here.
        if (!stmt->is<AllocaStmt>() && !stmt->is<AdStackAllocaStmt>() &&
            !stmt->is<ExternalFuncCallStmt>() &&
            !may_contain_variable(live_in_this_node, store_ptr) &&
            (contain_variable(killed_in_this_node, store_ptr) ||
             !may_contain_variable(live_out, store_ptr))) {
          // Neither used in other nodes nor used in this node.
          if (!stmt->is<AtomicOpStmt>()) {
            // Eliminate the dead store.
            erase(i);
            modified = true;
            continue;
          }
          auto atomic = stmt->cast<AtomicOpStmt>();
          // Weaken the atomic operation to a load.
          if (atomic->dest->is<AllocaStmt>()) {
            auto local_load = Stmt::make<LocalLoadStmt>(atomic->dest);
            local_load->ret_type = atomic->ret_type;
            // Notice that we have a load here
            // (the return value of AtomicOpStmt).
            live_in_this_node.insert(atomic->dest);
            live_load_in_this_node[atomic->dest] = local_load.get();
            killed_in_this_node.erase(atomic->dest);
            replace_with(i, std::move(local_load), true);
            modified = true;
            continue;
          } else if (!is_parallel_executed ||
                     (atomic->dest->is<GlobalPtrStmt>() &&
                      atomic->dest->as<GlobalPtrStmt>()->snode->is_scalar())) {
            // If this node is parallel executed, we can't weaken a global
            // atomic operation to a global load.
            // TODO: we can weaken it if it's element-wise (i.e. never
            //  accessed by other threads).
            auto global_load = Stmt::make<GlobalLoadStmt>(atomic->dest);
            global_load->ret_type = atomic->ret_type;
            // Notice that we have a load here
            // (the return value of AtomicOpStmt).
            live_in_this_node.insert(atomic->dest);
            live_load_in_this_node[atomic->dest] = global_load.get();
            killed_in_this_node.erase(atomic->dest);
            replace_with(i, std::move(global_load), true);
            modified = true;
            continue;
          }
        } else {
          // A non-eliminated store.
          killed_in_this_node.insert(store_ptr);
          auto old_live_in_this_node = std::move(live_in_this_node);
          live_in_this_node.clear();
          for (auto &var : old_live_in_this_node) {
            if (!irpass::analysis::definitely_same_address(store_ptr, var))
              live_in_this_node.insert(var);
          }
        }
      }
    }
    auto load_ptrs = irpass::analysis::get_load_pointers(stmt);
    if (load_ptrs.size() == 1 && store_ptrs.empty()) {
      // Identical load elimination
      auto load_ptr = load_ptrs.front();
      if (!after_lower_access ||
          (load_ptr->is<AllocaStmt>() || load_ptr->is<AdStackAllocaStmt>())) {
        // After lower_access, we only analyze local variables and stacks.
        if (live_load_in_this_node.find(load_ptr) !=
                live_load_in_this_node.end() &&
            !may_contain_variable(killed_in_this_node, load_ptr)) {
          // Only perform identical load elimination within a CFGNode.
          auto next_load_stmt = live_load_in_this_node[load_ptr];
          TI_ASSERT(irpass::analysis::same_statements(stmt, next_load_stmt));
          next_load_stmt->replace_usages_with(stmt);
          erase(block->locate(next_load_stmt));
          modified = true;
        }
        live_load_in_this_node[load_ptr] = stmt;
        killed_in_this_node.erase(load_ptr);
      }
    }
    for (auto &load_ptr : load_ptrs) {
      if (!after_lower_access ||
          (load_ptr->is<AllocaStmt>() || load_ptr->is<AdStackAllocaStmt>())) {
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

CFGNode *ControlFlowGraph::back() const {
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
    if (!nodes[i]->reach_in.empty()) {
      std::vector<std::string> indices;
      for (auto stmt : nodes[i]->reach_in) {
        indices.push_back(stmt->name());
      }
      node_info += fmt::format("; reach_in={{{}}}", fmt::join(indices, ", "));
    }
    if (!nodes[i]->reach_out.empty()) {
      std::vector<std::string> indices;
      for (auto stmt : nodes[i]->reach_out) {
        indices.push_back(stmt->name());
      }
      node_info += fmt::format("; reach_out={{{}}}", fmt::join(indices, ", "));
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
  for (int i = 0; i < num_nodes; i++) {
    for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
      auto stmt = nodes[i]->block->statements[j].get();
      if ((stmt->is<PtrOffsetStmt>() &&
           stmt->as<PtrOffsetStmt>()->origin->is<AllocaStmt>()) ||
          (!after_lower_access &&
           (stmt->is<GlobalPtrStmt>() || stmt->is<ExternalPtrStmt>() ||
            stmt->is<BlockLocalPtrStmt>() || stmt->is<ThreadLocalPtrStmt>() ||
            stmt->is<GlobalTemporaryStmt>() || stmt->is<PtrOffsetStmt>()))) {
        // TODO: unify them
        // A global pointer that may contain some data before this kernel.
        nodes[start_node]->reach_gen.insert(stmt);
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

  // The worklist algorithm.
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
      auto store_ptrs = irpass::analysis::get_store_destination(stmt);
      bool killed;
      if (store_ptrs.empty()) {  // the case of a global pointer
        killed = now->reach_kill_variable(stmt);
      } else {
        killed = true;
        for (auto store_ptr : store_ptrs) {
          if (!now->reach_kill_variable(store_ptr)) {
            killed = false;
            break;
          }
        }
      }
      if (!killed) {
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

void ControlFlowGraph::live_variable_analysis(
    bool after_lower_access,
    const std::optional<LiveVarAnalysisConfig> &config_opt) {
  TI_AUTO_PROF;
  const int num_nodes = size();
  std::queue<CFGNode *> to_visit;
  std::unordered_map<CFGNode *, bool> in_queue;
  TI_ASSERT(nodes[final_node]->empty());
  nodes[final_node]->live_gen.clear();
  nodes[final_node]->live_kill.clear();

  auto in_final_node_live_gen = [&config_opt](const Stmt *stmt) -> bool {
    if (stmt->is<AllocaStmt>() || stmt->is<AdStackAllocaStmt>()) {
      return false;
    }
    if (stmt->is<PtrOffsetStmt>() &&
        stmt->cast<PtrOffsetStmt>()->origin->is<AllocaStmt>()) {
      return false;
    }
    if (auto *gptr = stmt->cast<GlobalPtrStmt>();
        gptr && config_opt.has_value()) {
      const bool res = (config_opt->eliminable_snodes.count(gptr->snode) == 0);
      return res;
    }
    // A global pointer that may be loaded after this kernel.
    return true;
  };
  if (!after_lower_access) {
    for (int i = 0; i < num_nodes; i++) {
      for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
        auto stmt = nodes[i]->block->statements[j].get();
        for (auto store_ptr : irpass::analysis::get_store_destination(stmt)) {
          if (in_final_node_live_gen(store_ptr)) {
            nodes[final_node]->live_gen.insert(store_ptr);
          }
        }
      }
    }
  }
  for (int i = num_nodes - 1; i >= 0; i--) {
    // push into the queue in reversed order to make it slightly faster
    if (i != final_node) {
      nodes[i]->live_variable_analysis(after_lower_access);
    }
    nodes[i]->live_out.clear();
    nodes[i]->live_in = nodes[i]->live_gen;
    to_visit.push(nodes[i].get());
    in_queue[nodes[i].get()] = true;
  }

  // The worklist algorithm.
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
      // If a node is empty with in-degree or out-degree <= 1, we can eliminate
      // it (except for the start node and the final node).
      if (nodes[i] && nodes[i]->empty() && i != start_node && i != final_node &&
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
      if (final_node == i) {
        final_node = new_num_nodes;
      }
      new_num_nodes++;
    }
  }
  nodes.resize(new_num_nodes);
}

bool ControlFlowGraph::unreachable_code_elimination() {
  // Note that container statements are not in the control-flow graph, so
  // this pass cannot eliminate container statements properly for now.
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
      if (!node->empty()) {
        while (!node->empty())
          node->erase(node->end_location - 1);
        modified = true;
      }
    }
  }
  return modified;
}

bool ControlFlowGraph::store_to_load_forwarding(bool after_lower_access,
                                                bool autodiff_enabled) {
  TI_AUTO_PROF;
  reaching_definition_analysis(after_lower_access);
  const int num_nodes = size();
  bool modified = false;
  for (int i = 0; i < num_nodes; i++) {
    if (nodes[i]->store_to_load_forwarding(after_lower_access,
                                           autodiff_enabled))
      modified = true;
  }
  return modified;
}

bool ControlFlowGraph::dead_store_elimination(
    bool after_lower_access,
    const std::optional<LiveVarAnalysisConfig> &lva_config_opt) {
  TI_AUTO_PROF;
  live_variable_analysis(after_lower_access, lva_config_opt);
  const int num_nodes = size();
  bool modified = false;
  for (int i = 0; i < num_nodes; i++) {
    if (nodes[i]->dead_store_elimination(after_lower_access))
      modified = true;
  }
  return modified;
}

std::unordered_set<SNode *> ControlFlowGraph::gather_loaded_snodes() {
  TI_AUTO_PROF;
  reaching_definition_analysis(/*after_lower_access=*/false);
  const int num_nodes = size();
  std::unordered_set<SNode *> snodes;

  // Note: since global store may only partially modify a value state, the
  // result (which contains the modified and unmodified part) actually needs a
  // read from the previous version of the value state.
  //
  // I.e.,
  // output_value_state = merge(input_value_state, written_part)
  //
  // Therefore we include the nodes[final_node]->reach_in in snodes.
  for (auto &stmt : nodes[final_node]->reach_in) {
    if (auto global_ptr = stmt->cast<GlobalPtrStmt>()) {
      snodes.insert(global_ptr->snode);
    }
  }

  for (int i = 0; i < num_nodes; i++) {
    if (i != final_node) {
      nodes[i]->gather_loaded_snodes(snodes);
    }
  }
  return snodes;
}

void ControlFlowGraph::determine_ad_stack_size(int default_ad_stack_size) {
  /**
   * Determine all adaptive AD-stacks' necessary size using the Bellman-Ford
   * algorithm. When there is a positive loop (#pushes > #pops in a loop)
   * for an AD-stack, we cannot determine the size of the AD-stack, and
   * |default_ad_stack_size| is used. The time complexity is
   * O(num_statements + num_stacks * num_edges * num_nodes).
   */
  const int num_nodes = size();

  // max_increased_size[i][j] is the maximum number of (pushes - pops) of
  // stack |i| among all prefixes of the CFGNode |j|.
  std::unordered_map<AdStackAllocaStmt *, std::vector<int>> max_increased_size;

  // increased_size[i][j] is the number of (pushes - pops) of stack |i| in
  // the CFGNode |j|.
  std::unordered_map<AdStackAllocaStmt *, std::vector<int>> increased_size;

  std::unordered_map<CFGNode *, int> node_ids;
  std::unordered_set<AdStackAllocaStmt *> all_stacks;
  std::unordered_set<AdStackAllocaStmt *> indeterminable_stacks;

  for (int i = 0; i < num_nodes; i++)
    node_ids[nodes[i].get()] = i;

  for (int i = 0; i < num_nodes; i++) {
    for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
      Stmt *stmt = nodes[i]->block->statements[j].get();
      if (auto *stack = stmt->cast<AdStackAllocaStmt>()) {
        all_stacks.insert(stack);
        max_increased_size.insert(
            std::make_pair(stack, std::vector<int>(num_nodes, 0)));
        increased_size.insert(
            std::make_pair(stack, std::vector<int>(num_nodes, 0)));
      }
    }
  }

  // For each basic block we compute the increase of stack size. This is a
  // pre-processing step for the next maximum stack size determining algorithm.
  for (int i = 0; i < num_nodes; i++) {
    for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
      Stmt *stmt = nodes[i]->block->statements[j].get();
      if (auto *stack_push = stmt->cast<AdStackPushStmt>()) {
        auto *stack = stack_push->stack->as<AdStackAllocaStmt>();
        if (stack->max_size == 0 /*adaptive*/) {
          increased_size[stack][i]++;
          if (increased_size[stack][i] > max_increased_size[stack][i]) {
            max_increased_size[stack][i] = increased_size[stack][i];
          }
        }
      } else if (auto *stack_pop = stmt->cast<AdStackPopStmt>()) {
        auto *stack = stack_pop->stack->as<AdStackAllocaStmt>();
        if (stack->max_size == 0 /*adaptive*/) {
          increased_size[stack][i]--;
        }
      }
    }
  }

  // The maximum stack size determining algorithm -- run the Bellman-Ford
  // algorithm on each AD-stack separately.
  for (auto *stack : all_stacks) {
    // The maximum size of |stack| among all control flows starting at the
    // beginning of the IR.
    int max_size = 0;

    // max_size_at_node_begin[j] is the maximum size of |stack| among
    // all control flows starting at the beginning of the IR and ending at the
    // beginning of the CFGNode |j|. Initialize this array to -1 to make sure
    // that the first iteration of the Bellman-Ford algorithm fully updates
    // this array.
    std::vector<int> max_size_at_node_begin(num_nodes, -1);

    // The queue for the Bellman-Ford algorithm.
    std::queue<int> to_visit;

    // An optimization for the Bellman-Ford algorithm.
    std::vector<bool> in_queue(num_nodes);

    // An array for detecting positive loop in the Bellman-Ford algorithm.
    std::vector<int> times_pushed_in_queue(num_nodes, 0);

    max_size_at_node_begin[start_node] = 0;
    to_visit.push(start_node);
    in_queue[start_node] = true;
    times_pushed_in_queue[start_node]++;

    bool has_positive_loop = false;

    // The Bellman-Ford algorithm.
    while (!to_visit.empty()) {
      int node_id = to_visit.front();
      to_visit.pop();
      in_queue[node_id] = false;
      CFGNode *now = nodes[node_id].get();

      // Inside this CFGNode -- update the answer |max_size|
      const auto max_size_inside_this_node = max_increased_size[stack][node_id];
      const auto current_max_size =
          max_size_at_node_begin[node_id] + max_size_inside_this_node;
      if (current_max_size > max_size) {
        max_size = current_max_size;
      }
      // At the end of this CFGNode -- update the state
      // |max_size_at_node_begin| of other CFGNodes
      const auto increase_in_this_node = increased_size[stack][node_id];
      const auto current_size =
          max_size_at_node_begin[node_id] + increase_in_this_node;
      for (auto *next_node : now->next) {
        int next_node_id = node_ids[next_node];
        if (current_size > max_size_at_node_begin[next_node_id]) {
          max_size_at_node_begin[next_node_id] = current_size;
          if (!in_queue[next_node_id]) {
            if (times_pushed_in_queue[next_node_id] <= num_nodes) {
              to_visit.push(next_node_id);
              in_queue[next_node_id] = true;
              times_pushed_in_queue[next_node_id]++;
            } else {
              // A positive loop is found because a node is going to be pushed
              // into the queue the (num_nodes + 1)-th time.
              has_positive_loop = true;
              break;
            }
          }
        }
      }
      if (has_positive_loop) {
        break;
      }
    }

    if (has_positive_loop) {
      stack->max_size = default_ad_stack_size;
      indeterminable_stacks.insert(stack);
    } else {
      // Since we use |max_size| == 0 for adaptive sizes, we do not want stacks
      // with maximum capacity indeed equal to 0.
      TI_WARN_IF(max_size == 0,
                 "Unused autodiff stack {} should have been eliminated.",
                 stack->name());
      stack->max_size = max_size;
    }
  }

  // Print a debug message if we have indeterminable AD-stacks' sizes.
  if (!indeterminable_stacks.empty()) {
    std::vector<std::string> indeterminable_stacks_name;
    indeterminable_stacks_name.reserve(indeterminable_stacks.size());
    for (auto &stack : indeterminable_stacks) {
      indeterminable_stacks_name.push_back(stack->name());
    }
    TI_DEBUG(
        "Unable to determine the necessary size for autodiff stacks [{}]. "
        "Use "
        "configured size (CompileConfig::default_ad_stack_size) {} instead.",
        fmt::join(indeterminable_stacks_name, ", "), default_ad_stack_size);
  }
}

}  // namespace lang
}  // namespace taichi
