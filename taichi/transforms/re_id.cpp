#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

// This pass manipulates the id of statements so that they are successive values
// starting from 0
class ReId : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  int id_counter;

  ReId() : id_counter(0) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void re_id(Stmt *stmt) {
    stmt->id = id_counter++;
  }

  void visit(Stmt *stmt) override {
    re_id(stmt);
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    re_id(stmt);
  }

  static void run(IRNode *node) {
    ReId instance;
    node->accept(&instance);
  }
};

namespace irpass {
void re_id(IRNode *root) {
  ReId::run(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
