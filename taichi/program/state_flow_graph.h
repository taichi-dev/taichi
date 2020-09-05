#pragma once

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/program/async_utils.h"

TLANG_NAMESPACE_BEGIN

class StateFlowGraph {
 public:
  struct Node;
  using StateToNodeMappnig =
      std::unordered_map<AsyncState, Node *, AsyncStateHash>;
  // Each node is a task
  // Note: after SFG is done, each node here should hold a TaskLaunchRecord.
  // Optimization should happen fully on the SFG, instead of the queue in
  // AsyncEngine.
  // Before we migrate, the SFG is only used for visualization. Therefore we
  // only store a kernel_name, which is used as the label of the GraphViz node.
  struct Node {
    //  TODO: make use of IRHandle here
    IRNode *root;
    std::string kernel_name;
    StateToNodeMappnig input_edges, output_edges;
  };

  StateToNodeMappnig latest_state_owner;

  std::vector<std::unique_ptr<Node>> nodes;

  Node *initial_node;  // The initial node holds all the initial states.

  StateFlowGraph() {
    nodes.push_back(std::make_unique<Node>());
    initial_node = nodes.back().get();
    initial_node->kernel_name = "initial_state";
  }

  void print_edges(const StateToNodeMappnig &edges);

  void print();

  void dump_dot(const std::string &fn) {
    // TODO: export the graph to Dot format for GraphViz
  }

  void insert_task(TaskMeta task_meta);

  void insert_state_flow(Node *from, Node *to, AsyncState state);
};

TLANG_NAMESPACE_END
