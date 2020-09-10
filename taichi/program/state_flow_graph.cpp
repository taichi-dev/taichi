#include "taichi/program/state_flow_graph.h"
#include "taichi/util/bit.h"

#include <sstream>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

std::string StateFlowGraph::Node::string() const {
  return fmt::format("[node: {}:{}]", task_name, launch_id);
}

StateFlowGraph::StateFlowGraph() {
  nodes_.push_back(std::make_unique<Node>());
  initial_node_ = nodes_.back().get();
  initial_node_->task_name = "initial_state";
  initial_node_->launch_id = 0;
}

void StateFlowGraph::clear() {
  // TODO: GC here
  nodes_.resize(1);  // Erase all nodes except the initial one
  initial_node_->output_edges.clear();
  latest_state_owner_.clear();

  // Do not clear task_name_to_launch_ids_.
}

void StateFlowGraph::insert_task(const TaskLaunchRecord &rec,
                                 const TaskMeta &task_meta) {
  auto node = std::make_unique<Node>();
  node->rec = rec;
  node->task_name = task_meta.kernel_name;
  {
    int &id = task_name_to_launch_ids_[node->task_name];
    node->launch_id = id;
    ++id;
  }
  for (auto input_state : task_meta.input_states) {
    if (latest_state_owner_.find(input_state) == latest_state_owner_.end()) {
      latest_state_owner_[input_state] = initial_node_;
    }
    insert_state_flow(latest_state_owner_[input_state], node.get(),
                      input_state);
  }
  for (auto output_state : task_meta.output_states) {
    latest_state_owner_[output_state] = node.get();
  }
  nodes_.push_back(std::move(node));
}

void StateFlowGraph::insert_state_flow(Node *from, Node *to, AsyncState state) {
  TI_ASSERT(from != nullptr);
  TI_ASSERT(to != nullptr);
  from->output_edges[state].insert(to);
  to->input_edges.insert(std::make_pair(state, from));
}

bool StateFlowGraph::fuse() {
  using SFGNode = StateFlowGraph::Node;
  using bit::Bitset;
  const int n = nodes_.size();
  for (int i = 0; i < n; i++) {
    nodes_[i]->node_id = i;
  }

  // Compute the transitive closure.
  auto has_path = std::make_unique<Bitset[]>(n);
  auto has_path_reverse = std::make_unique<Bitset[]>(n);
  // has_path[i][j] denotes if there is a path from i to j.
  // has_path_reverse[i][j] denotes if there is a path from j to i.
  for (int i = 0; i < n; i++) {
    has_path[i] = Bitset(n);
    has_path[i][i] = true;
    has_path_reverse[i] = Bitset(n);
    has_path_reverse[i][i] = true;
  }
  for (int i = 0; i < n; i++) {
    for (auto &edges : nodes_[i]->output_edges) {
      for (auto &edge : edges.second) {
        // Assume nodes are sorted in topological order.
        TI_ASSERT(edge->node_id > i);
        has_path[edge->node_id] |= has_path[i];
      }
    }
  }
  for (int i = n - 1; i >= 0; i--) {
    for (auto &edge : nodes_[i]->input_edges) {
      has_path_reverse[edge.second->node_id] |= has_path_reverse[i];
    }
  }

  return false;
}

std::vector<TaskLaunchRecord> StateFlowGraph::extract() {
  std::vector<TaskLaunchRecord> tasks;
  tasks.reserve(nodes_.size());
  for (int i = 1; i < (int)nodes_.size(); i++) {
    tasks.push_back(nodes_[i]->rec);
  }
  clear();
  return tasks;
}

void StateFlowGraph::print() {
  fmt::print("=== State Flow Graph ===\n");
  for (auto &node : nodes_) {
    fmt::print("{}\n", node->string());
    if (!node->input_edges.empty()) {
      fmt::print("  Inputs:\n");
      for (const auto &p : node->input_edges) {
        fmt::print("    {} <- {}\n", p.first.name(), p.second->string());
      }
    }
    if (!node->output_edges.empty()) {
      fmt::print("  Outputs:\n");
      for (const auto &p : node->output_edges) {
        for (const auto *to : p.second) {
          fmt::print("    {} -> {}\n", p.first.name(), to->string());
        }
      }
    }
  }
  fmt::print("=======================\n");
}

std::string StateFlowGraph::dump_dot() {
  using SFGNode = StateFlowGraph::Node;
  std::stringstream ss;
  ss << "digraph {\n";
  auto node_id = [](const SFGNode *n) {
    // https://graphviz.org/doc/info/lang.html ID naming
    return fmt::format("n_{}_{}", n->task_name, n->launch_id);
  };
  // Specify the node styles
  std::unordered_set<const SFGNode *> latest_state_nodes;
  for (const auto &p : latest_state_owner_) {
    latest_state_nodes.insert(p.second);
  }
  std::vector<const SFGNode *> nodes_with_no_inputs;
  for (const auto &nd : nodes_) {
    const auto *n = nd.get();
    ss << "  " << fmt::format("{} [label=\"{}\"", node_id(n), n->string());
    if (n == initial_node_) {
      ss << ",shape=box";
    } else if (latest_state_nodes.find(n) != latest_state_nodes.end()) {
      ss << ",peripheries=2";
    }
    ss << "]\n";
    if (nd->input_edges.empty())
      nodes_with_no_inputs.push_back(n);
  }
  ss << "\n";
  {
    // DFS
    std::unordered_set<const SFGNode *> visited;
    std::vector<const SFGNode *> stack(nodes_with_no_inputs);
    while (!stack.empty()) {
      auto *from = stack.back();
      stack.pop_back();
      if (visited.find(from) == visited.end()) {
        visited.insert(from);
        for (const auto &p : from->output_edges) {
          for (const auto *to : p.second) {
            stack.push_back(to);

            ss << "  "
               << fmt::format("{} -> {} [label=\"{}\"]", node_id(from),
                              node_id(to), p.first.name())
               << '\n';
          }
        }
      }
    }
  }
  ss << "}\n";  // closes "dirgraph {"
  return ss.str();
}

TLANG_NAMESPACE_END
