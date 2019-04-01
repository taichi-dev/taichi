#include "../ir.h"
#include "../scratch_pad.h"

TLANG_NAMESPACE_BEGIN

// Figure out accessed snodes, and their ranges in this for stmt
class AccessAnalysis : public IRVisitor {
 public:
  StructForStmt *for_stmt;

  AccessAnalysis(StructForStmt *for_stmt) : for_stmt(for_stmt) {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;

    const auto &block = for_stmt->body;

    for (int i = 0; i < (int)block->statements.size(); i++) {
      block->statements[i]->accept(this);
    }
  }

  void visit(GlobalPtrStmt *stmt) override {
  }

  // Do not eliminate global data access
  void visit(GlobalLoadStmt *stmt) override {
    TC_ASSERT(stmt->width() == 1);  // TODO: support
    for (int l = 0; l < stmt->width(); l++) {
      auto ptr_ = stmt->ptr;
      auto ptr = ptr_->as<GlobalPtrStmt>();
      auto snode = ptr->snodes[l];
      bool matching_indices = true;
      std::vector<int> offsets;  // TODO: change to offset_ranges
      offsets.resize(ptr->indices.size());
      int num_indices = (int)ptr->indices.size();
      for (int i = 0; i < num_indices; i++) {
        auto diff =
            analysis::value_diff(ptr->indices[i], l, for_stmt->loop_vars[i]);
        if (diff.first) {
          offsets[i] = diff.second;
        } else {
          matching_indices = false;
        }
      }
      if (matching_indices) {
        TC_INFO("Detected regular access");
        for (int i = 0; i < num_indices; i++) {
          TC_P(offsets[i]);
        }
      }
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    return;
  }

  void visit(AtomicOpStmt *stmt) override {
    return;
  }

  void visit(Stmt *stmt) override {
    TC_ASSERT(!stmt->is_container_statement());
  }
};

class InsertScratchPad : public IRVisitor {
 public:
  InsertScratchPad(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    node->accept(this);
  }

  void visit(Block *block) override {
    for (auto &stmt : block->statements) {
      stmt->accept(this);
    }
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(RangeForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    // do the work here...
    TC_P(for_stmt->cached_level);
    if (for_stmt->cached_level != -1) {
      AccessAnalysis _(for_stmt);
      // WeakenAccess _(for_stmt);
    }
    for_stmt->body->accept(this);
  }
};

namespace irpass {

void insert_scratch_pad(IRNode *root) {
  InsertScratchPad _(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
