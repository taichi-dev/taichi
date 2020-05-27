#include "taichi/ir/cfg.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

// Build a control-flow graph
class CFGBuilder : public IRVisitor {
 private:
  std::unique_ptr<ControlFlowGraph> graph;
  Block *current_block;
  CFGNode *last_node_in_current_block;
  std::vector<CFGNode *> continues_in_current_loop;
  std::vector<CFGNode *> breaks_in_current_loop;
  int current_stmt_id;
  int begin_location;
  std::vector<CFGNode *> prev_nodes;

 public:
  CFGBuilder()
      : current_block(nullptr),
        last_node_in_current_block(nullptr),
        current_stmt_id(-1),
        begin_location(-1) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    graph = std::make_unique<ControlFlowGraph>();
  }

  void visit(Stmt *stmt) override {
    if (stmt->is_container_statement()) {
      TI_ERROR("Visitor for container stmt undefined.");
    }
  }

  CFGNode *new_node() {
    auto node = graph->push_back(current_block, begin_location,
                                 current_stmt_id - 1,
                                 last_node_in_current_block);
    for (auto &prev_node : prev_nodes) {
      CFGNode::add_edge(prev_node, node);
    }
    prev_nodes.clear();
    begin_location = current_stmt_id + 1;
    last_node_in_current_block = node;
    return node;
  }

  void visit(ContinueStmt *stmt) override {
    continues_in_current_loop.push_back(new_node());
  }

  void visit(WhileControlStmt *stmt) override {
    auto node = new_node();
    breaks_in_current_loop.push_back(node);
    prev_nodes.push_back(node);
  }

  void visit(IfStmt *if_stmt) override {
    auto before_if = new_node();
    begin_location = -1;
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

  void visit_loop(Block *body, CFGNode *before_loop, bool is_while_true) {
    int loop_stmt_id = current_stmt_id;
    auto backup_continues = std::move(continues_in_current_loop);
    auto backup_breaks = std::move(breaks_in_current_loop);
    begin_location = -1;
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
    visit_loop(stmt->body.get(), new_node(), true);
  }

  void visit(RangeForStmt *stmt) override {
    visit_loop(stmt->body.get(), new_node(), false);
  }

  void visit(StructForStmt *stmt) override {
    visit_loop(stmt->body.get(), new_node(), false);
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->has_body()) {
      auto before_offload = new_node();
      if (stmt->task_type == OffloadedStmt::TaskType::struct_for ||
          stmt->task_type == OffloadedStmt::TaskType::range_for) {
        visit_loop(stmt->body.get(), before_offload, false);
      } else {
        TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::serial);
        int offload_stmt_id = current_stmt_id;
        auto block_begin_index = graph->size();
        begin_location = -1;
        stmt->body->accept(this);
        prev_nodes.push_back(graph->back());
        // Container statements don't belong to any CFGNodes.
        begin_location = offload_stmt_id + 1;
        CFGNode::add_edge(before_offload,
                          graph->nodes[block_begin_index].get());
      }
    }
  }

  void visit(Block *block) override {
    auto backup_block = current_block;
    auto backup_last_node = last_node_in_current_block;
    TI_ASSERT(begin_location == -1);
    TI_ASSERT(prev_nodes.empty());
    current_block = block;
    last_node_in_current_block = nullptr;
    begin_location = 0;

    for (int i = 0; i < (int)block->size(); i++) {
      current_stmt_id = i;
      block->statements[i]->accept(this);
    }
    current_stmt_id = block->size();
    new_node();  // Each block has a deterministic last node.

    current_block = backup_block;
    last_node_in_current_block = backup_last_node;
    begin_location = -1;
  }

  static std::unique_ptr<ControlFlowGraph> run(IRNode *root) {
    CFGBuilder builder;
    root->accept(&builder);
    return std::move(builder.graph);
  }
};

namespace irpass::analysis {
std::unique_ptr<ControlFlowGraph> build_cfg(IRNode *root) {
  return CFGBuilder::run(root);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
