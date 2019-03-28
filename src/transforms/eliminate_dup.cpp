#include "../ir.h"

TLANG_NAMESPACE_BEGIN

class BasicBlockEliminate : public IRVisitor {
 public:
  Block *block;

  int current_stmt_id;

  BasicBlockEliminate(Block *block) : block(block) {
    // allow_undefined_visitor = true;
    // invoke_default_visitor = false;
    run();
  }

  void run() {
    for (int i = 0; i < (int)block->statements.size(); i++) {
      current_stmt_id = i;
      block->statements[i]->accept(this);
    }
  }

  void visit(GlobalPtrStmt *stmt) override {
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<GlobalPtrStmt>();
          bool same = true;
          for (int l = 0; l < stmt->width(); l++) {
            if (stmt->snode[l] != bstmt_->snode[l]) {
              same = false;
              break;
            }
          }
          if (stmt->indices.size() != bstmt_->indices.size()) {
            same = false;
          } else {
            for (int j = 0; j < (int)stmt->indices.size(); j++) {
              if (stmt->indices[j] != bstmt_->indices[j])
                same = false;
            }
          }
          if (same) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModifiedException();
          }
        }
      }
    }
  }

  void visit(ConstStmt *stmt) override {
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      auto &bstmt_data = *bstmt;
      if (typeid(bstmt_data) == typeid(*stmt)) {
        if (stmt->width() == bstmt->width()) {
          auto bstmt_ = bstmt->as<ConstStmt>();
          bool same = true;
          for (int l = 0; l < stmt->width(); l++) {
            if (!stmt->val[l].equal_type_and_value(bstmt_->val[l])) {
              same = false;
              break;
            }
          }
          if (same) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModifiedException();
          }
        }
      }
    }
  }

  void visit(AllocaStmt *stmt) override {
    return;
  }

  void visit(ElementShuffleStmt *stmt) override {
    // is this stmt necessary?
    {
      bool same_source = true;
      bool inc_index = true;
      for (int l = 0; l < stmt->width(); l++) {
        if (stmt->elements[l].stmt != stmt->elements[0].stmt)
          same_source = false;
        if (stmt->elements[l].index != l)
          inc_index = false;
      }
      if (same_source && inc_index &&
          stmt->elements[0].stmt->ret_type == stmt->ret_type) {
        // useless shuffle.
        stmt->replace_with(stmt->elements[0].stmt);
        stmt->parent->erase(current_stmt_id);
        throw IRModifiedException();
      }
    }

    // find dup
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<ElementShuffleStmt>();
          bool same = true;
          for (int l = 0; l < stmt->width(); l++) {
            if (stmt->elements[l].stmt != bstmt_->elements[l].stmt ||
                stmt->elements[l].index != bstmt_->elements[l].index) {
              same = false;
              break;
            }
          }
          if (same) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModifiedException();
          }
        }
      }
    }
  }

  void visit(LocalLoadStmt *stmt) override {
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<LocalLoadStmt>();
          bool same = true;
          std::vector<Stmt *> vars;
          for (int l = 0; l < stmt->width(); l++) {
            vars.push_back(stmt->ptr[l].var);
            if (stmt->ptr[l].var != bstmt_->ptr[l].var ||
                stmt->ptr[l].offset != bstmt_->ptr[l].offset) {
              same = false;
              break;
            }
          }
          if (same) {
            // no store to the var?
            bool has_related_store = false;
            for (int j = i + 1; j < current_stmt_id; j++) {
              if (block->statements[j]
                      ->is_container_statement()) {  // no if, while, etc..
                has_related_store = true;
                TC_WARN("Jumping");
                break;
              }
              if (block->statements[j]->is<LocalStoreStmt>()) {
                auto st = block->statements[j]->as<LocalStoreStmt>();
                for (auto var : vars) {
                  if (st->ptr == var) {
                    has_related_store = true;
                    break;
                  }
                }
              }
            }
            if (!has_related_store) {
              stmt->replace_with(bstmt.get());
              stmt->parent->erase(current_stmt_id);
              throw IRModifiedException();
            }
          }
        }
      }
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    return;
  }

  // Do not eliminate global data access
  void visit(GlobalLoadStmt *stmt) override {
    return;
  }

  void visit(GlobalStoreStmt *stmt) override {
    return;
  }

  void visit(UnaryOpStmt *stmt) override {
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<UnaryOpStmt>();
          if (bstmt_->same_operation(stmt) && bstmt_->rhs == stmt->rhs) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModifiedException();
          }
        }
      }
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<BinaryOpStmt>();
          if (bstmt_->op_type == stmt->op_type && bstmt_->lhs == stmt->lhs &&
              bstmt_->rhs == stmt->rhs) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModifiedException();
          }
        }
      }
    }
  }

  void visit(TrinaryOpStmt *stmt) override {
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<TrinaryOpStmt>();
          if (bstmt_->op_type == stmt->op_type && bstmt_->op1 == stmt->op1 &&
              bstmt_->op2 == stmt->op2 && bstmt_->op3 == stmt->op3) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModifiedException();
          }
        }
      }
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    return;
  }

  void visit(PrintStmt *stmt) override {
    return;
  }

  void visit(RandStmt *stmt) override {
    return;
  }

  void visit(WhileControlStmt *stmt) override {
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

  void visit(Block *block) override {
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

  void visit(RangeForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }
};

namespace irpass {

void eliminate_dup(IRNode *root) {
  EliminateDup _(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
