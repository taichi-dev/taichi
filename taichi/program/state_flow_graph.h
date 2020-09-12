#pragma once

#include <unordered_map>
#include <unordered_set>

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/program/async_utils.h"

TLANG_NAMESPACE_BEGIN

class StateFlowGraph {
 public:
  struct Node;
  using StateToNodeMapping =
      std::unordered_map<AsyncState, Node *, AsyncStateHash>;

  // Each node is a task
  // Note: after SFG is done, each node here should hold a TaskLaunchRecord.
  // Optimization should happen fully on the SFG, instead of the queue in
  // AsyncEngine.
  // Before we migrate, the SFG is only used for visualization. Therefore we
  // only store a kernel_name, which is used as the label of the GraphViz node.
  struct Node {
    TaskLaunchRecord rec;
    std::string task_name;
    // Incremental ID to identify the i-th launch of the task.
    int launch_id;

    StateToNodeMapping input_edges;
    // Profiling showed horrible performance using std::unordered_multimap (at
    // least on Mac with clang-1103.0.32.62)...
    std::unordered_map<AsyncState, std::unordered_set<Node *>, AsyncStateHash>
        output_edges;

    std::string string() const;
  };

  StateFlowGraph();

  void print();

  std::string dump_dot();

  void insert_task(const TaskLaunchRecord &rec, const TaskMeta &task_meta);

  void insert_state_flow(Node *from, Node *to, AsyncState state);

 private:
  std::vector<std::unique_ptr<Node>> nodes_;
  Node *initial_node_;  // The initial node holds all the initial states.
  StateToNodeMapping latest_state_owner_;
  std::unordered_map<std::string, int> task_name_to_launch_ids_;
};

TLANG_NAMESPACE_END
