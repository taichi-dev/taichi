#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/function.h"

namespace taichi {
namespace lang {

struct CFGFuncKey {
  FunctionKey func_key{"", -1, -1};
  bool in_parallel_for{false};

  bool operator==(const CFGFuncKey &other_key) const {
    return func_key == other_key.func_key &&
           in_parallel_for == other_key.in_parallel_for;
  }
};

}  // namespace lang
}  // namespace taichi

namespace std {
template <>
struct hash<taichi::lang::CFGFuncKey> {
  std::size_t operator()(const taichi::lang::CFGFuncKey &key) const noexcept {
    return std::hash<taichi::lang::FunctionKey>()(key.func_key) ^
           ((std::size_t)key.in_parallel_for << 32);
  }
};
}  // namespace std

namespace taichi {
namespace lang {

/**
 * Build a control-flow graph. The resulting graph is guaranteed to have an
 * empty start node and an empty final node.
 *
 * In the following docstrings, node... means a CFGNode's corresponding
 * statements in the CHI IR. Other blocks are just Blocks in the CHI IR.
 * Nodes denoted with "()" mean not yet created when visiting the Stmt/Block.
 *
 * Structures like
 * node_a {
 *   ...
 * } -> node_b, node_c;
 * means node_a has edges to node_b and node_c, or equivalently, node_b and
 * node_c appear in the |next| field of node_a.
 *
 * Structures like
 * node_a {
 *   ...
 * } -> node_b, [node_c if "cond"];
 * means node_a has an edge to node_b, and node_a has an edge to node_b iff
 * the condition "cond" is true.
 *
 * When there can be many CFGNodes in a Block, internal nodes are omitted for
 * simplicity.
 *
 * TODO(#2193): Make sure ReturnStmt is handled properly.
 */
class CFGBuilder : public IRVisitor {
 public:
  CFGBuilder()
      : current_block(nullptr),
        last_node_in_current_block(nullptr),
        current_stmt_id(-1),
        begin_location(-1),
        current_offload(nullptr),
        in_parallel_for(false) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    graph = std::make_unique<ControlFlowGraph>();
    // Make an empty start node.
    auto start_node = graph->push_back();
    prev_nodes.push_back(start_node);
  }

  void visit(Stmt *stmt) override {
    if (stmt->is_container_statement()) {
      TI_ERROR("Visitor for container statement undefined.");
    }
  }

  /**
   * Create a node for the current control-flow graph,
   * mark the current statement as the end location (exclusive) of the node,
   * and add edges from |prev_nodes| to the node.
   *
   * @param next_begin_location The location in the IR block of the first
   * statement in the next node, if the next node is in the same IR block of
   * the node to be returned. Otherwise, next_begin_location must be -1.
   * @return The node which is just created.
   */
  CFGNode *new_node(int next_begin_location) {
    auto node = graph->push_back(
        current_block, begin_location, /*end_location=*/current_stmt_id,
        /*is_parallel_executed=*/in_parallel_for,
        /*prev_node_in_same_block=*/last_node_in_current_block);
    for (auto &prev_node : prev_nodes) {
      // Now that the "(next node)" is created, we should insert edges
      // "node... -> (next node)" here.
      CFGNode::add_edge(prev_node, node);
    }
    prev_nodes.clear();
    begin_location = next_begin_location;
    last_node_in_current_block = node;
    return node;
  }

  /**
   * Structure:
   *
   * block {
   *   node {
   *     ...
   *   } -> node_loop_begin, (the next node after the loop);
   *   continue;
   *   (next node) {
   *     ...
   *   }
   * }
   *
   * Note that the edges are inserted in visit_loop().
   */
  void visit(ContinueStmt *stmt) override {
    // Don't put ContinueStmt in any CFGNodes.
    continues_in_current_loop.push_back(new_node(current_stmt_id + 1));
  }

  /**
   * Structure:
   *
   * block {
   *   node {
   *     ...
   *   } -> (next node), (the next node after the loop);
   *   while_control (possibly break);
   *   (next node) {
   *     ...
   *   }
   * }
   *
   * Note that the edges are inserted in visit_loop().
   */
  void visit(WhileControlStmt *stmt) override {
    // Don't put WhileControlStmt in any CFGNodes.
    auto node = new_node(current_stmt_id + 1);
    breaks_in_current_loop.push_back(node);
    prev_nodes.push_back(node);
  }

  /**
   * Structure:
   *
   * block {
   *   node {
   *     ...
   *   } -> node_func_begin;
   *   foo();
   *   (next node) {
   *     ...
   *   }
   * }
   *
   * foo() {
   *   node_func_begin {
   *     ...
   *   } -> ... -> node_func_end;
   *   node_func_end {
   *     ...
   *   } -> (next node);
   * }
   */
  void visit(FuncCallStmt *stmt) override {
    auto node_before_func_call = new_node(-1);
    CFGFuncKey func_key = {stmt->func->func_key, in_parallel_for};
    if (node_func_begin.count(func_key) == 0) {
      // Generate CFG for the function.
      TI_ASSERT(stmt->func->ir->is<Block>());
      auto func_begin_index = graph->size();
      stmt->func->ir->accept(this);
      node_func_begin[func_key] = graph->nodes[func_begin_index].get();
      node_func_end[func_key] = graph->nodes.back().get();
    }
    CFGNode::add_edge(node_before_func_call, node_func_begin[func_key]);
    prev_nodes.push_back(node_func_end[func_key]);

    // Don't put FuncCallStmt in any CFGNodes.
    begin_location = current_stmt_id + 1;
  }

  /**
   * Structure:
   *
   * node_before_if {
   *   ...
   * } -> node_true_branch_begin, node_false_branch_begin;
   * if (...) {
   *   node_true_branch_begin {
   *     ...
   *   } -> ... -> node_true_branch_end;
   *   node_true_branch_end {
   *     ...
   *   } -> (next node);
   * } else {
   *   node_false_branch_begin {
   *     ...
   *   } -> ... -> node_false_branch_end;
   *   node_false_branch_end {
   *     ...
   *   } -> (next node);
   * }
   * (next node) {
   *   ...
   * }
   */
  void visit(IfStmt *if_stmt) override {
    auto before_if = new_node(-1);
    CFGNode *true_branch_end = nullptr;
    if (if_stmt->true_statements) {
      auto true_branch_begin = graph->size();
      if_stmt->true_statements->accept(this);
      CFGNode::add_edge(before_if, graph->nodes[true_branch_begin].get());
      true_branch_end = graph->back();
    }
    CFGNode *false_branch_end = nullptr;
    if (if_stmt->false_statements) {
      auto false_branch_begin = graph->size();
      if_stmt->false_statements->accept(this);
      CFGNode::add_edge(before_if, graph->nodes[false_branch_begin].get());
      false_branch_end = graph->back();
    }
    TI_ASSERT(prev_nodes.empty());
    if (if_stmt->true_statements)
      prev_nodes.push_back(true_branch_end);
    if (if_stmt->false_statements)
      prev_nodes.push_back(false_branch_end);
    if (!if_stmt->true_statements || !if_stmt->false_statements)
      prev_nodes.push_back(before_if);
    // Container statements don't belong to any CFGNodes.
    begin_location = current_stmt_id + 1;
  }

  /**
   * Structure ([(next node) if !is_while_true] means the node has an edge to
   * (next node) only when is_while_true is false):
   *
   * node_before_loop {
   *   ...
   * } -> node_loop_begin, [(next node) if !is_while_true];
   * loop (...) {
   *   node_loop_begin {
   *     ...
   *   } -> ... -> node_loop_end;
   *   node_loop_end {
   *     ...
   *   } -> node_loop_begin, [(next node) if !is_while_true];
   * }
   * (next node) {
   *   ...
   * }
   */
  void visit_loop(Block *body, CFGNode *before_loop, bool is_while_true) {
    int loop_stmt_id = current_stmt_id;
    auto backup_continues = std::move(continues_in_current_loop);
    auto backup_breaks = std::move(breaks_in_current_loop);
    continues_in_current_loop.clear();
    breaks_in_current_loop.clear();

    auto loop_begin_index = graph->size();
    body->accept(this);
    auto loop_begin = graph->nodes[loop_begin_index].get();
    CFGNode::add_edge(before_loop, loop_begin);
    auto loop_end = graph->back();
    CFGNode::add_edge(loop_end, loop_begin);
    if (!is_while_true) {
      prev_nodes.push_back(before_loop);
      prev_nodes.push_back(loop_end);
    }
    for (auto &node : continues_in_current_loop) {
      CFGNode::add_edge(node, loop_begin);
      prev_nodes.push_back(node);
    }
    for (auto &node : breaks_in_current_loop) {
      prev_nodes.push_back(node);
    }

    // Container statements don't belong to any CFGNodes.
    begin_location = loop_stmt_id + 1;
    continues_in_current_loop = std::move(backup_continues);
    breaks_in_current_loop = std::move(backup_breaks);
  }

  void visit(WhileStmt *stmt) override {
    visit_loop(stmt->body.get(), new_node(-1), true);
  }

  void visit(RangeForStmt *stmt) override {
    auto old_in_parallel_for = in_parallel_for;
    if (!current_offload)
      in_parallel_for = true;
    visit_loop(stmt->body.get(), new_node(-1), false);
    in_parallel_for = old_in_parallel_for;
  }

  void visit(StructForStmt *stmt) override {
    auto old_in_parallel_for = in_parallel_for;
    if (!current_offload)
      in_parallel_for = true;
    visit_loop(stmt->body.get(), new_node(-1), false);
    in_parallel_for = old_in_parallel_for;
  }

  /**
   * Structure:
   *
   * node_before_offload {
   *   ...
   * } -> node_tls_prologue;
   * node_tls_prologue {
   *   ...
   * } -> node_bls_prologue;
   * node_bls_prologue {
   *   ...
   * } -> node_body;
   * node_body {
   *   ...
   * } -> node_bls_epilogue;
   * node_bls_epilogue {
   *   ...
   * } -> node_tls_epilogue;
   * node_tls_epilogue {
   *   ...
   * } -> (next node);
   * (next node) {
   *   ...
   * }
   */
  void visit(OffloadedStmt *stmt) override {
    current_offload = stmt;
    if (stmt->tls_prologue) {
      auto before_offload = new_node(-1);
      int offload_stmt_id = current_stmt_id;
      auto block_begin_index = graph->size();
      stmt->tls_prologue->accept(this);
      prev_nodes.push_back(graph->back());
      // Container statements don't belong to any CFGNodes.
      begin_location = offload_stmt_id + 1;
      CFGNode::add_edge(before_offload, graph->nodes[block_begin_index].get());
    }
    if (stmt->bls_prologue) {
      auto before_offload = new_node(-1);
      int offload_stmt_id = current_stmt_id;
      auto block_begin_index = graph->size();
      stmt->bls_prologue->accept(this);
      prev_nodes.push_back(graph->back());
      // Container statements don't belong to any CFGNodes.
      begin_location = offload_stmt_id + 1;
      CFGNode::add_edge(before_offload, graph->nodes[block_begin_index].get());
    }
    if (stmt->has_body()) {
      auto before_offload = new_node(-1);
      int offload_stmt_id = current_stmt_id;
      auto block_begin_index = graph->size();
      if (stmt->task_type == OffloadedStmt::TaskType::range_for ||
          stmt->task_type == OffloadedStmt::TaskType::struct_for) {
        in_parallel_for = true;
      }
      stmt->body->accept(this);
      in_parallel_for = false;
      prev_nodes.push_back(graph->back());
      // Container statements don't belong to any CFGNodes.
      begin_location = offload_stmt_id + 1;
      CFGNode::add_edge(before_offload, graph->nodes[block_begin_index].get());
    }
    if (stmt->bls_epilogue) {
      auto before_offload = new_node(-1);
      int offload_stmt_id = current_stmt_id;
      auto block_begin_index = graph->size();
      stmt->bls_epilogue->accept(this);
      prev_nodes.push_back(graph->back());
      // Container statements don't belong to any CFGNodes.
      begin_location = offload_stmt_id + 1;
      CFGNode::add_edge(before_offload, graph->nodes[block_begin_index].get());
    }
    if (stmt->tls_epilogue) {
      auto before_offload = new_node(-1);
      int offload_stmt_id = current_stmt_id;
      auto block_begin_index = graph->size();
      stmt->tls_epilogue->accept(this);
      prev_nodes.push_back(graph->back());
      // Container statements don't belong to any CFGNodes.
      begin_location = offload_stmt_id + 1;
      CFGNode::add_edge(before_offload, graph->nodes[block_begin_index].get());
    }
    current_offload = nullptr;
  }

  /**
   * Structure:
   *
   * graph->start_node {
   *   // no statements
   * } -> node_block_begin if this is the first top-level block;
   * block {
   *   node_block_begin {
   *     ...
   *   } -> ... -> node_block_end;
   *   node_block_end {
   *     ...
   *   }
   * }
   *
   * graph->final_node = node_block_end;
   */
  void visit(Block *block) override {
    auto backup_block = current_block;
    auto backup_last_node = last_node_in_current_block;
    auto backup_stmt_id = current_stmt_id;
    // |begin_location| must be -1 (indicating we are not building any CFGNode)
    // when the |current_block| changes.
    TI_ASSERT(begin_location == -1);
    TI_ASSERT(prev_nodes.empty() || graph->size() == 1);
    current_block = block;
    last_node_in_current_block = nullptr;
    begin_location = 0;

    for (int i = 0; i < (int)block->size(); i++) {
      current_stmt_id = i;
      block->statements[i]->accept(this);
    }
    current_stmt_id = block->size();
    new_node(-1);  // Each block has a deterministic last node.
    graph->final_node = (int)graph->size() - 1;

    current_block = backup_block;
    last_node_in_current_block = backup_last_node;
    current_stmt_id = backup_stmt_id;
  }

  static std::unique_ptr<ControlFlowGraph> run(IRNode *root) {
    CFGBuilder builder;
    root->accept(&builder);
    if (!builder.graph->nodes[builder.graph->final_node]->empty()) {
      // Make the final node empty (by adding an empty final node).
      builder.graph->push_back();
      CFGNode::add_edge(builder.graph->nodes[builder.graph->final_node].get(),
                        builder.graph->back());
      builder.graph->final_node = (int)builder.graph->size() - 1;
    }
    return std::move(builder.graph);
  }

 private:
  std::unique_ptr<ControlFlowGraph> graph;
  Block *current_block;
  CFGNode *last_node_in_current_block;
  std::vector<CFGNode *> continues_in_current_loop;
  std::vector<CFGNode *> breaks_in_current_loop;
  int current_stmt_id;
  int begin_location;
  std::vector<CFGNode *> prev_nodes;
  OffloadedStmt *current_offload;
  bool in_parallel_for;
  std::unordered_map<CFGFuncKey, CFGNode *> node_func_begin;
  std::unordered_map<CFGFuncKey, CFGNode *> node_func_end;
};

namespace irpass::analysis {
std::unique_ptr<ControlFlowGraph> build_cfg(IRNode *root) {
  return CFGBuilder::run(root);
}
}  // namespace irpass::analysis

}  // namespace lang
}  // namespace taichi
