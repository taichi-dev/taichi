
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/lang_util.h"
#include "taichi/program/context.h"
#include "taichi/program/async_utils.h"

TLANG_NAMESPACE_BEGIN

class StateFlowGraph {
 public:
  // TODO: maybe unordered_map is better?
  std::map<AsyncState, Node *> latest_state_owner;

  // Each node is a task
  // Note: after SFG is done, each node here should hold a TaskLaunchRecord.
  // Optimization should happen fully on the SFG, instead of the queue in
  // AsyncEngine.
  // Before we migrate, the SFG is only used for visualization. Therefore we
  // only store a kernel_name.
  struct Node {
    //  TODO: make use of IRHandle here
    IRNode *root;
    std::string kernel_name;
    std::unordered_map<AsyncState, Node *> input_edges, output_edges;
  };

  std::vector<std::unique_ptr<Node>> nodes;

  StateFlowGraph() {
  }

  void export(const std::string &fn) {
    // TODO: export the graph to Dot format for GraphViz
  }

  Node *insert_task();

  void insert_state_flow(Node *from, Node *to, AsyncState state);
};

TLANG_NAMESPACE_END
