#include "taichi/program/state_flow_graph.h"

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
  from->output_edges.insert(std::make_pair(state, to));
  to->input_edges.insert(std::make_pair(state, from));
}

bool StateFlowGraph::fuse() {
  return false;
}

std::vector<TaskLaunchRecord> StateFlowGraph::extract() {
  std::vector<TaskLaunchRecord> tasks;
  return tasks;
}

void StateFlowGraph::print_edges(const StateFlowGraph::Edges &edges) {
  for (auto &edge : edges) {
    auto input_node = edge.second;
    fmt::print("    {} -> {}\n", edge.first.name(), input_node->string());
  }
}

void StateFlowGraph::print() {
  fmt::print("=== State Flow Graph ===\n");
  for (auto &node : nodes_) {
    fmt::print("{}\n", node->string());
    if (!node->input_edges.empty()) {
      fmt::print("  Inputs:\n");
      print_edges(node->input_edges);
    }
    if (!node->output_edges.empty()) {
      fmt::print("  Outputs:\n");
      print_edges(node->output_edges);
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
          auto *to = p.second;
          stack.push_back(to);

          ss << "  "
             << fmt::format("{} -> {} [label=\"{}\"]", node_id(from),
                            node_id(to), p.first.name())
             << '\n';
        }
      }
    }
  }
  ss << "}\n";  // closes "dirgraph {"
  return ss.str();
}

TLANG_NAMESPACE_END
