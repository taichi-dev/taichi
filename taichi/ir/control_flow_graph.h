#pragma once

#include <optional>
#include <unordered_set>

#include "taichi/ir/ir.h"

namespace taichi {
namespace lang {

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
  // For accelerating get_store_forwarding_data()
  std::unordered_set<Block *> parent_blocks_;

  mutable std::unordered_map<Stmt *, int> location_cache_;
  int locate_in_block(Stmt *stmt) const;

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

  // Property methods.
  bool empty() const;
  std::size_t size() const;

  // Methods for modifying the underlying CHI IR.
  void erase(int location);
  void insert(std::unique_ptr<Stmt> &&new_stmt, int location);
  void replace_with(int location,
                    std::unique_ptr<Stmt> &&new_stmt,
                    bool replace_usages = true) const;

  // Utility methods.
  static bool contain_variable(const std::unordered_set<Stmt *> &var_set,
                               Stmt *var);
  static bool may_contain_variable(const std::unordered_set<Stmt *> &var_set,
                                   Stmt *var);
  bool reach_kill_variable(Stmt *var) const;
  Stmt *get_store_forwarding_data(Stmt *var, int position) const;

  // Analyses and optimizations inside a CFGNode.
  void reaching_definition_analysis(bool after_lower_access);
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

  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] CFGNode *back() const;

  void print_graph_structure() const;

  /**
   * Perform reaching definition analysis using the worklist algorithm,
   * and store the results in CFGNodes.
   * https://en.wikipedia.org/wiki/Reaching_definition
   *
   * @param after_lower_access
   *   When after_lower_access is true, only consider local variables (allocas).
   */
  void reaching_definition_analysis(bool after_lower_access);

  /**
   * Perform live variable analysis using the worklist algorithm,
   * and store the results in CFGNodes.
   * https://en.wikipedia.org/wiki/Live_variable_analysis
   *
   * @param after_lower_access
   *   When after_lower_access is true, only consider local variables (allocas).
   * @param config_opt
   *   The set of SNodes which is never loaded after this task.
   */
  void live_variable_analysis(
      bool after_lower_access,
      const std::optional<LiveVarAnalysisConfig> &config_opt);

  /**
   * Simplify the graph structure to accelerate other analyses and
   * optimizations. The IR is not modified.
   */
  void simplify_graph();

  // This pass cannot eliminate container statements properly for now.
  bool unreachable_code_elimination();

  /**
   * Perform store-to-load forwarding and identical store elimination.
   */
  bool store_to_load_forwarding(bool after_lower_access);

  /**
   * Perform dead store elimination and identical load elimination.
   */
  bool dead_store_elimination(
      bool after_lower_access,
      const std::optional<LiveVarAnalysisConfig> &lva_config_opt);

  /**
   * Gather the SNodes which is read or partially written in this offloaded
   * task.
   */
  std::unordered_set<SNode *> gather_loaded_snodes();

  /**
   * Determine all adaptive AD-stacks' necessary size.
   * @param default_ad_stack_size The default AD-stack's size when we are
   * unable to determine some AD-stack's size.
   */
  void determine_ad_stack_size(int default_ad_stack_size);
};

}  // namespace lang
}  // namespace taichi
