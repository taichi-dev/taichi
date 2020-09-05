#include "taichi/program/state_flow_graph.h"

TLANG_NAMESPACE_BEGIN

void StateFlowGraph::insert_task(const TaskMeta &task_meta) {
  auto node = std::make_unique<Node>();
  node->kernel_name = task_meta.kernel_name;
  for (auto input_state : task_meta.input_states) {
    if (latest_state_owner.find(input_state) == latest_state_owner.end()) {
      latest_state_owner[input_state] = initial_node;
    }
    insert_state_flow(latest_state_owner[input_state], node.get(), input_state);
  }
  for (auto output_state : task_meta.output_states) {
    latest_state_owner[output_state] = node.get();
  }
  nodes.push_back(std::move(node));
}

void StateFlowGraph::insert_state_flow(Node *from, Node *to, AsyncState state) {
  TI_ASSERT(from != nullptr);
  TI_ASSERT(to != nullptr);
  from->output_edges.insert(std::make_pair(state, to));
  to->input_edges.insert(std::make_pair(state, from));
}

void StateFlowGraph::print_edges(
    const StateFlowGraph::StateToNodeMapping &edges) {
  for (auto &edge : edges) {
    auto input_node = edge.second;
    fmt::print("    {}: node {} @ {}\n", edge.first.name(),
               input_node->kernel_name, (void *)input_node);
  }
}

void StateFlowGraph::print() {
  fmt::print("=== State Flow Graph ===\n");
  for (auto &node : nodes) {
    fmt::print("Node {} {}\n", node->kernel_name, (void *)node.get());
    if (!node->input_edges.empty()) {
      fmt::print("  Inputs:\n", node->kernel_name, (void *)node.get());
      print_edges(node->input_edges);
    }
    if (!node->output_edges.empty()) {
      fmt::print("  Outputs:\n", node->kernel_name, (void *)node.get());
      print_edges(node->output_edges);
    }
  }
  fmt::print("=======================\n");
}

TLANG_NAMESPACE_END
