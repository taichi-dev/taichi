#pragma once

#include <optional>
#include <unordered_set>

#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

/**
 * A basic block in control-flow graph.
 * A CFGNode contains a reference to a part of the CHI IR, or more precisely,
 * an interval of statements in a Block.
 * The edges in the graph are stored in |prev| and |next|. The control flow is
 * possible to go from any node in |prev| to this node, and is possible to go
 * from this node to any node in |next|.
 */
class CFGNode {
 private:
  // For accelerating get_store_forwarding_data
  std::unordered_set<Block *> parent_blocks;

 public:
  // This node corresponds to block->statements[i]
  // for i in [begin_location, end_location).
  Block *block;
  int begin_location, end_location;
  // Is this node in an offloaded range_for/struct_for?
  bool is_parallel_executed;

  // For updating begin/end locations when modifying the block.
  CFGNode *prev_node_in_same_block;
  CFGNode *next_node_in_same_block;

  // Edges in the graph
  std::vector<CFGNode *> prev, next;

  // Reaching definition analysis
  // https://en.wikipedia.org/wiki/Reaching_definition
  std::unordered_set<Stmt *> reach_gen, reach_kill, reach_in, reach_out;

  // Live variable analysis
  // https://en.wikipedia.org/wiki/Live_variable_analysis
  std::unordered_set<Stmt *> live_gen, live_kill, live_in, live_out;

  CFGNode(Block *block,
          int begin_location,
          int end_location,
          bool is_parallel_executed,
          CFGNode *prev_node_in_same_block);

  // An empty node
  CFGNode();

  static void add_edge(CFGNode *from, CFGNode *to);
  bool empty() const;
  std::size_t size() const;
  void erase(int location);
  void insert(std::unique_ptr<Stmt> &&new_stmt, int location);
  void replace_with(int location,
                    std::unique_ptr<Stmt> &&new_stmt,
                    bool replace_usages = true);

  static bool contain_variable(const std::unordered_set<Stmt *> &var_set,
                               Stmt *var);
  static bool may_contain_variable(const std::unordered_set<Stmt *> &var_set,
                                   Stmt *var);
  void reaching_definition_analysis(bool after_lower_access);
  bool reach_kill_variable(Stmt *var) const;
  Stmt *get_store_forwarding_data(Stmt *var, int position) const;
  bool store_to_load_forwarding(bool after_lower_access);
  void gather_loaded_snodes(std::unordered_set<SNode *> &snodes) const;

  void live_variable_analysis(bool after_lower_access);
  bool dead_store_elimination(bool after_lower_access);
};

class ControlFlowGraph {
 private:
  // Erase an empty node.
  void erase(int node_id);

 public:
  struct LiveVarAnalysisConfig {
    // This is mostly useful for SFG task-level dead store elimination. SFG may
    // detect certain cases where writes to one or more SNodes in a task are
    // eliminable.
    std::unordered_set<const SNode *> eliminable_snodes;
  };
  std::vector<std::unique_ptr<CFGNode>> nodes;
  const int start_node = 0;
  int final_node{0};

  template <typename... Args>
  CFGNode *push_back(Args &&... args) {
    nodes.emplace_back(std::make_unique<CFGNode>(std::forward<Args>(args)...));
    return nodes.back().get();
  }

  std::size_t size() const;
  CFGNode *back();

  void print_graph_structure() const;
  void reaching_definition_analysis(bool after_lower_access);
  void live_variable_analysis(
      bool after_lower_access,
      const std::optional<LiveVarAnalysisConfig> &config_opt);

  void simplify_graph();

  // This pass cannot eliminate container statements properly for now.
  bool unreachable_code_elimination();

  // Also performs identical store elimination.
  bool store_to_load_forwarding(bool after_lower_access);

  // Also performs identical load elimination.
  bool dead_store_elimination(
      bool after_lower_access,
      const std::optional<LiveVarAnalysisConfig> &lva_config_opt);

  // Gather the SNodes this offload reads.
  std::unordered_set<SNode *> gather_loaded_snodes();
};

TLANG_NAMESPACE_END
