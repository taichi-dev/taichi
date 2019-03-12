#include "../ir.h"

TLANG_NAMESPACE_BEGIN

class BasicBlockEliminate : public IRVisitor {
 public:
  Block *block;

  BasicBlockEliminate(Block *block) : block(block) {
    // allow_undefined_visitor = true;
    // invoke_default_visitor = false;
    run();
  }

  void run() {
    for (int i = 0; i < (int)block->statements.size(); i++) {
      block->statements[i]->accept(this);
    }
  }

  void visit(GlobalPtrStmt *stmt) {
    TC_NOT_IMPLEMENTED
  }

  void visit(ConstStmt *stmt) {
    TC_NOT_IMPLEMENTED
  }

  void visit(AllocaStmt *stmt) {
    return;
  }

  void visit(ElementShuffleStmt *stmt) {
    TC_NOT_IMPLEMENTED
  }

  void visit(LocalLoadStmt *stmt) {
    TC_NOT_IMPLEMENTED
  }

  void visit(LocalStoreStmt *stmt) {
    return;
  }

  // Do not eliminate global data access
  void visit(GlobalLoadStmt *stmt) {
    return;
  }

  void visit(GlobalStoreStmt *stmt) {
    return;
  }

  void visit(BinaryOpStmt *stmt) {
    TC_NOT_IMPLEMENTED
  }

  void visit(UnaryOpStmt *stmt) {
    TC_NOT_IMPLEMENTED
  }

  void visit(PrintStmt *stmt) {
    return;
  }

  void visit(RandStmt *stmt) {
    return;
  }

  void visit(WhileControlStmt *stmt) {
    return;
  }
};

class EliminateDup : public IRVisitor {
 public:
  EliminateDup(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    node->accept(this);
  }

  void visit(Block *block) {
    if (!block->has_container_statements()) {
      while (true) {
        try {
          BasicBlockEliminate _(block);
        } catch (IRModifiedException) {
          continue;
        }
        break;
      }
    } else {
      for (auto &stmt : block->statements) {
        stmt->accept(this);
      }
    }
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(RangeForStmt *for_stmt) {
    auto old_vectorize = for_stmt->vectorize;
    for_stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }
};

namespace irpass {

void eliminate_dup(IRNode *root) {
  EliminateDup _(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
