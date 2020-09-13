#include "taichi/program/state_flow_graph.h"

#include <sstream>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// TODO: rename state to edge since we have not only state flow edges but also
// dependency edges.

std::string StateFlowGraph::Node::string() const {
  return fmt::format("[node: {}:{}]", task_name, launch_id);
}

StateFlowGraph::StateFlowGraph() {
  nodes_.push_back(std::make_unique<Node>());
  initial_node_ = nodes_.back().get();
  initial_node_->task_name = "initial_state";
  initial_node_->launch_id = 0;
  initial_node_->is_initial_node = true;
}

void StateFlowGraph::insert_task(const TaskLaunchRecord &rec,
                                 const TaskMeta &task_meta) {
  auto node = std::make_unique<Node>();
  node->rec = rec;
  node->task_name = task_meta.kernel_name;
  node->input_states = task_meta.input_states;
  node->output_states = task_meta.output_states;
  node->task_type = task_meta.type;
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
    if (latest_state_readers_.find(output_state) ==
        latest_state_readers_.end()) {
      latest_state_readers_[output_state].insert(initial_node_);
    }
    for (auto &d : latest_state_readers_[output_state]) {
      // insert a dependency edge
      insert_state_flow(d, node.get(), output_state);
    }
    latest_state_readers_[output_state].clear();
  }

  // Note that this loop must happen AFTER the previous one
  for (auto input_state : task_meta.input_states) {
    latest_state_readers_[input_state].insert(node.get());
  }
  nodes_.push_back(std::move(node));
}

void StateFlowGraph::insert_state_flow(Node *from, Node *to, AsyncState state) {
  TI_ASSERT(from != nullptr);
  TI_ASSERT(to != nullptr);
  from->output_edges[state].insert(to);
  to->input_edges[state].insert(from);
}

void StateFlowGraph::print() {
  fmt::print("=== State Flow Graph ===\n");
  for (auto &node : nodes_) {
    fmt::print("{}\n", node->string());
    if (!node->input_edges.empty()) {
      fmt::print("  Inputs:\n");
      for (const auto &p : node->input_edges) {
        for (const auto *to : p.second) {
          fmt::print("    {} <- {}\n", p.first.name(), to->string());
        }
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

std::string StateFlowGraph::dump_dot(
    const std::optional<std::string> &rankdir) {
  using SFGNode = StateFlowGraph::Node;
  using TaskType = OffloadedStmt::TaskType;
  std::stringstream ss;
  ss << "digraph {\n";
  auto node_id = [](const SFGNode *n) {
    // https://graphviz.org/doc/info/lang.html ID naming
    return fmt::format("n_{}_{}", n->task_name, n->launch_id);
  };
  // Graph level configuration.
  if (rankdir) {
    ss << "  rankdir=" << *rankdir << "\n";
  }
  ss << "\n";
  // Specify the node styles
  std::unordered_set<const SFGNode *> latest_state_nodes;
  for (const auto &p : latest_state_owner_) {
    latest_state_nodes.insert(p.second);
  }
  std::vector<const SFGNode *> nodes_with_no_inputs;
  for (const auto &nd : nodes_) {
    const auto *n = nd.get();
    ss << "  " << fmt::format("{} [label=\"{}\"", node_id(n), n->string());
    if (nd->is_initial_node) {
      ss << ",shape=box";
    } else if (latest_state_nodes.find(n) != latest_state_nodes.end()) {
      ss << ",peripheries=2";
    }
    // Highlight user-defined tasks
    const auto tt = nd->task_type;
    if (!nd->is_initial_node &&
        (tt == TaskType::range_for || tt == TaskType::struct_for ||
         tt == TaskType::serial)) {
      ss << ",style=filled,fillcolor=lightgray";
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
            std::string style;

            if (!from->has_state_flow(p.first, to)) {
              style = "style=dotted";
            }

            ss << "  "
               << fmt::format("{} -> {} [label=\"{}\" {}]", node_id(from),
                              node_id(to), p.first.name(), style)
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
