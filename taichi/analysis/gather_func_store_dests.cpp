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
  struct TarjanData {
    std::unordered_map<Function *, int> func_dfn;
    std::unordered_map<Function *, int> func_low;
    std::unordered_set<Function *> func_in_stack;
    std::stack<Function *> func_stack;
  };
  TarjanData &tarjan_data_;

  static std::unordered_set<Stmt *> run(Function *func,
                                        TarjanData &tarjan_data) {
    TI_ASSERT(tarjan_data.func_dfn.count(func) == 0);
    tarjan_data.func_dfn[func] = tarjan_data.func_low[func] =
        tarjan_data.func_dfn.size();
    tarjan_data.func_in_stack.insert(func);
    tarjan_data.func_stack.push(func);
    GatherFuncStoreDests searcher(func, tarjan_data);
    func->ir->accept(&searcher);
    if (tarjan_data.func_low[func] == tarjan_data.func_dfn[func]) {
      while (true) {
        auto top = tarjan_data.func_stack.top();
        tarjan_data.func_stack.pop();
        tarjan_data.func_in_stack.erase(top);
        top->store_dests.insert(searcher.results_.begin(),
                                searcher.results_.end());
        if (top == func) {
          break;
        }
      }
    }
    return searcher.results_;
  }

  static void run(IRNode *ir, TarjanData &tarjan_data) {
    GatherFuncStoreDests searcher(nullptr, tarjan_data);
    ir->accept(&searcher);
  }

 public:
  using BasicStmtVisitor::visit;

  GatherFuncStoreDests(Function *func, TarjanData &tarjan_data)
      : current_func_(func), tarjan_data_(tarjan_data) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    if (!current_func_) {
      return;
    }
    auto result = irpass::analysis::get_store_destination(stmt);
    for (const auto &dest : result) {
      if (dest->is<ExternalPtrStmt>()) {
        continue;
      }
      if (auto matrix_ptr = dest->cast<MatrixPtrStmt>()) {
        if (matrix_ptr->origin->is<ExternalPtrStmt>()) {
          continue;
        }
      }
      results_.insert(dest);
    }
  }

  void visit(FuncCallStmt *stmt) override {
    auto func = stmt->func;
    if (!current_func_) {
      if (!tarjan_data_.func_dfn.count(func)) {
        run(func, tarjan_data_);
      }
      return;
    }
    if (!tarjan_data_.func_dfn.count(func)) {
      auto result = run(func, tarjan_data_);
      results_.merge(result);
      tarjan_data_.func_low[current_func_] = std::min(
          tarjan_data_.func_low[current_func_], tarjan_data_.func_low[func]);
    } else if (tarjan_data_.func_in_stack.count(func)) {
      tarjan_data_.func_low[current_func_] = std::min(
          tarjan_data_.func_low[current_func_], tarjan_data_.func_dfn[func]);
    } else {
      const auto &dests = func->store_dests;
      results_.insert(dests.begin(), dests.end());
    }
  }

  static void run(IRNode *ir) {
    TarjanData tarjan_data;
    run(ir, tarjan_data);
  }
};

namespace irpass::analysis {
void gather_func_store_dests(IRNode *ir) {
  GatherFuncStoreDests::run(ir);
}

}  // namespace irpass::analysis

}  // namespace taichi::lang
