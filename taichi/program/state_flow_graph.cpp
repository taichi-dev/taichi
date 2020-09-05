#include "taichi/program/state_flow_graph.h"

TLANG_NAMESPACE_BEGIN

Node *StateFlowGraph::insert_task() {
  std::unique_ptr<Node> node;
  nodes.push_back(std::move(node));
}

void StateFlowGraph::insert_state_flow(Node *from, Node *to, AsyncState state) {
  from->output_edges.insert(state, to);
  to->input_edges.insert(state, from);
}

TLANG_NAMESPACE_END
