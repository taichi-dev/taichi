#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

class StmtSearcher : public BasicStmtVisitor {
 private:
  std::function<bool(Stmt *)> test;
  std::vector<Stmt *> results;

 public:
  using BasicStmtVisitor::visit;

  StmtSearcher(std::function<bool(Stmt *)> test) : test(test) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) {
    if (test(stmt))
      results.push_back(stmt);
  }

  static std::vector<Stmt *> run(IRNode *root,
                                 const std::function<bool(Stmt *)> &test) {
    StmtSearcher searcher(test);
    root->accept(&searcher);
    return searcher.results;
  }
};

namespace irpass::analysis {
std::vector<Stmt *> gather_statements(IRNode *root,
                                      const std::function<bool(Stmt *)> &test) {
  return StmtSearcher::run(root, test);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
