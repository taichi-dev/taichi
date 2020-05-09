#include "taichi/ir/cfg.h"
#include "taichi/ir/ir.h"

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
  CFGBuilder() : current_block(nullptr),
                 last_node_in_current_block(nullptr),
                 current_stmt_id(0),
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
    auto node = graph->push_back(current_block, begin_location, current_stmt_id - 1);
    for (auto &prev_node : prev_nodes) {
      CFGNode::add_edge(prev_node, node);
    }
    prev_nodes.clear();
    begin_location = current_stmt_id + 1;
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
      true_branch_end = graph->nodes.back().get();
    }
    CFGNode *false_branch_end = nullptr;
    if (if_stmt->false_statements) {
      auto false_branch_begin = graph->size();
      if_stmt->false_statements->accept(this);
      CFGNode::add_edge(before_if, graph->nodes[false_branch_begin].get());
      false_branch_end = graph->nodes.back().get();
    }
    TI_ASSERT(prev_nodes.empty());
    if (if_stmt->true_statements)
      prev_nodes.push_back(true_branch_end);
    if (if_stmt->false_statements)
      prev_nodes.push_back(false_branch_end);
    if (!if_stmt->true_statements || !if_stmt->false_statements)
      prev_nodes.push_back(before_if);
    begin_location = current_stmt_id + 1;
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
    new_node();

    current_block = backup_block;
    last_node_in_current_block = backup_last_node;
    begin_location = -1;
  }

  static ControlFlowGraph run(IRNode *root) {
    CFGBuilder builder;
    root->accept(&builder);
  }
};

TLANG_NAMESPACE_END
