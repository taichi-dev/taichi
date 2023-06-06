#include <stack>
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/program/function.h"

namespace taichi::lang {

class GatherFuncStoreDests : public BasicStmtVisitor {
 private:
  std::unordered_set<Stmt *> results_;
  Function *current_func_;
  ControlFlowGraph *graph_;
  struct TarjanData {
    std::unordered_map<Function *, int> func_dfn;
    std::unordered_map<Function *, int> func_low;
    std::unordered_set<Function *> func_in_stack;
    std::stack<Function *> func_stack;
  };
  TarjanData &tarjan_data_;

  static std::unordered_set<Stmt *> run(Function *func,
                                        ControlFlowGraph *graph,
                                        TarjanData &tarjan_data) {
    TI_ASSERT(tarjan_data.func_dfn.count(func) == 0);
    tarjan_data.func_dfn[func] = tarjan_data.func_low[func] =
        tarjan_data.func_dfn.size();
    tarjan_data.func_in_stack.insert(func);
    tarjan_data.func_stack.push(func);
    GatherFuncStoreDests searcher(func, graph, tarjan_data);
    func->ir->accept(&searcher);
    if (tarjan_data.func_low[func] == tarjan_data.func_dfn[func]) {
      while (true) {
        auto top = tarjan_data.func_stack.top();
        tarjan_data.func_stack.pop();
        tarjan_data.func_in_stack.erase(top);
        TI_ASSERT(graph->func_store_dests.count(top) == 0);
        graph->func_store_dests[top] = searcher.results_;
        if (top == func) {
          break;
        }
      }
    }
    return searcher.results_;
  }

 public:
  using BasicStmtVisitor::visit;

  GatherFuncStoreDests(Function *func,
                       ControlFlowGraph *graph,
                       TarjanData &tarjan_data)
      : current_func_(func), graph_(graph), tarjan_data_(tarjan_data) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    auto result = irpass::analysis::get_store_destination(stmt);
    results_.insert(result.begin(), result.end());
  }

  void visit(FuncCallStmt *stmt) override {
    auto func = stmt->func;
    if (!tarjan_data_.func_dfn.count(func)) {
      auto result = run(func, graph_, tarjan_data_);
      results_.merge(result);
      tarjan_data_.func_low[current_func_] = std::min(
          tarjan_data_.func_low[current_func_], tarjan_data_.func_low[func]);
    } else if (tarjan_data_.func_in_stack.count(func)) {
      tarjan_data_.func_low[current_func_] = std::min(
          tarjan_data_.func_low[current_func_], tarjan_data_.func_dfn[func]);
    } else {
      const auto &dests = graph_->func_store_dests.at(func);
      results_.insert(dests.begin(), dests.end());
    }
  }

  static const std::unordered_set<Stmt *> &run(Function *func,
                                               ControlFlowGraph *graph) {
    TarjanData tarjan_data;
    run(func, graph, tarjan_data);
    return graph->func_store_dests.at(func);
  }
};

namespace irpass::analysis {
const std::unordered_set<Stmt *> &gather_func_store_dests(
    Function *func,
    ControlFlowGraph *graph) {
  if (graph->func_store_dests.count(func)) {
    return graph->func_store_dests.at(func);
  }
  return GatherFuncStoreDests::run(func, graph);
}

}  // namespace irpass::analysis

}  // namespace taichi::lang
