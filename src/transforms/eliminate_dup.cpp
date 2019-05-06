#include <taichi/taichi>
#include <set>
#include "../ir.h"

TLANG_NAMESPACE_BEGIN

class BasicBlockEliminate : public IRVisitor {
 public:
  Block *block;

  int current_stmt_id;
  std::set<int> &visited;
  StructForStmt *current_struct_for;

  BasicBlockEliminate(Block *block,
                      std::set<int> &visited,
                      StructForStmt *current_struct_for)
      : block(block), visited(visited), current_struct_for(current_struct_for) {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
    current_struct_for = nullptr;
    run();
  }

  bool is_done(Stmt *stmt) {
    return visited.find(stmt->instance_id) != visited.end();
  }

  void set_done(Stmt *stmt) {
    visited.insert(stmt->instance_id);
  }

  void run() {
    for (int i = 0; i < (int)block->statements.size(); i++) {
      current_stmt_id = i;
      block->statements[i]->accept(this);
    }
  }

  void visit(Stmt *stmt) override {
    if (stmt->is_container_statement())
      return;
    else {
      TC_ERROR("Visitor for non-container stmt undefined.");
    }
  }

  void visit(GlobalPtrStmt *stmt) override {
    if (is_done(stmt))
      return;
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<GlobalPtrStmt>();
          bool same = true;
          for (int l = 0; l < stmt->width(); l++) {
            if (stmt->snodes[l] != bstmt_->snodes[l]) {
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
    set_done(stmt);
  }

  void visit(ConstStmt *stmt) override {
    if (is_done(stmt))
      return;
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
    set_done(stmt);
  }

  void visit(AllocaStmt *stmt) override {
    return;
  }

  void visit(ElementShuffleStmt *stmt) override {
    if (is_done(stmt))
      return;
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

    // weaken indexing
    if (current_struct_for && stmt->width() == 1) {
      auto &loop_vars = current_struct_for->loop_vars;
      for (int k = 0; k < (int)loop_vars.size(); k++) {
        auto diff = analysis::value_diff(stmt->elements[0].stmt,
                                         stmt->elements[0].index,
                                         current_struct_for->loop_vars[k]);
        if (diff.related && diff.certain()) {
          auto load = stmt->insert_before_me(
              Stmt::make<LocalLoadStmt>(LocalAddress(loop_vars[k], 0)));
          load->ret_type.data_type = DataType::i32;
          auto constant = stmt->insert_before_me(
              Stmt::make<ConstStmt>(TypedConstant(diff.low)));
          constant->ret_type.data_type = DataType::i32;
          auto add = stmt->insert_before_me(
              Stmt::make<BinaryOpStmt>(BinaryType::add, load, constant));
          add->ret_type.data_type = DataType::i32;
          stmt->replace_with(add);
          stmt->parent->erase(stmt);
          throw IRModifiedException();
        }
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
            stmt->parent->erase(stmt);
            throw IRModifiedException();
          }
        }
      }
    }
    set_done(stmt);
  }

  void visit(LocalLoadStmt *stmt) override {
    if (is_done(stmt))
      return;
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

    // store-forwarding
    bool regular = true;
    auto alloca = stmt->ptr[0].var;
    for (int l = 0; l < stmt->width(); l++) {
      if (stmt->ptr[l].offset != l || stmt->ptr[l].var != alloca) {
        regular = false;
      }
    }
    if (regular) {
      for (int i = current_stmt_id - 1; i >= 0; i--) {
        auto &bstmt = block->statements[i];
        if (bstmt->is<LocalStoreStmt>()) {
          auto bstmt_ = bstmt->as<LocalStoreStmt>();
          if (bstmt_->ptr == alloca) {
            // forward
            stmt->replace_with(bstmt_->data);
            stmt->parent->erase(current_stmt_id);
            throw IRModifiedException();
          }
        } else if (bstmt->is_container_statement()) {
          // assume this container may modify the local var
          break;
        }
      }
    }
    set_done(stmt);
  }

  void visit(LocalStoreStmt *stmt) override {
    return;
  }

  void visit(GlobalLoadStmt *stmt) override {
    if (is_done(stmt))
      return;
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<GlobalLoadStmt>();
          bool same = stmt->ptr == bstmt_->ptr;
          if (same) {
            // no store to the var?
            bool has_store = false;
            for (int j = i + 1; j < current_stmt_id; j++) {
              if (block->statements[j]
                      ->is_container_statement()) {  // no if, while, etc..
                has_store = true;
                break;
              }
              if (block->statements[j]->is<GlobalStoreStmt>()) {
                has_store = true;
              }
            }
            if (!has_store) {
              stmt->replace_with(bstmt.get());
              stmt->parent->erase(current_stmt_id);
              throw IRModifiedException();
            }
          }
        }
      }
    }
    set_done(stmt);
  }

  void visit(GlobalStoreStmt *stmt) override {
    return;
  }

  void visit(SNodeOpStmt *stmt) override {
    return;
  }

  void visit(UnaryOpStmt *stmt) override {
    if (is_done(stmt))
      return;
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
    set_done(stmt);
  }

  void visit(BinaryOpStmt *stmt) override {
    if (is_done(stmt))
      return;
    if ((stmt->op_type == BinaryType::add ||
         stmt->op_type == BinaryType::sub) &&
        stmt->ret_type.data_type == DataType::i32) {
      if (stmt->rhs->is<ConstStmt>()) {
        auto stmt_ = stmt->rhs->as<ConstStmt>();
        bool all_zero = true;
        for (int l = 0; l < stmt_->width(); l++) {
          if (stmt_->val[l].val_int32() != 0) {
            all_zero = false;
          }
        }
        if (all_zero) {
          stmt->replace_with(stmt->lhs);
          stmt->parent->erase(current_stmt_id);
          throw IRModifiedException();
        }
      }
    }
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
    set_done(stmt);
  }

  void visit(TrinaryOpStmt *stmt) override {
    if (is_done(stmt))
      return;
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
    set_done(stmt);
  }

  static int div_floor(int a, int b) {
    if (a < 0) {
      return (a - b + 1) / b;
    } else {
      return a / b;
    }
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override {
    if (is_done(stmt))
      return;
    // step 1: try weakening when a struct for is used
    if (current_struct_for && !stmt->simplified) {
      auto &loop_vars = current_struct_for->loop_vars;
      for (int k = 0; k < (int)loop_vars.size(); k++) {
        auto diff = analysis::value_diff(stmt->input, 0,
                                         current_struct_for->loop_vars[k]);
        if (diff.related && diff.certain()) {
          // case 1: last loop var, vectorized, has assumption on vec size
          if (k == (int)current_struct_for->loop_vars.size() - 1) {
            auto load = stmt->insert_before_me(
                Stmt::make<LocalLoadStmt>(LocalAddress(loop_vars[k], 0)));
            load->ret_type.data_type = DataType::i32;
            stmt->input = load;  // TODO: needs a DIE pass
            int bound = 1 << stmt->bit_end;
            stmt->offset = ((diff.low % bound + bound) % bound) &
                           ~((1 << (stmt->bit_begin)) - 1);
          } else {
            // insert constant
            auto load = stmt->insert_before_me(
                Stmt::make<LocalLoadStmt>(LocalAddress(loop_vars[k], 0)));
            load->ret_type.data_type = DataType::i32;
            auto constant = stmt->insert_before_me(
                Stmt::make<ConstStmt>(TypedConstant(diff.low)));
            auto add = stmt->insert_before_me(
                Stmt::make<BinaryOpStmt>(BinaryType::add, load, constant));
            add->ret_type.data_type = DataType::i32;
            stmt->input = add;
          }
          stmt->simplified = true;
          throw IRModifiedException();
        }
      }
    }

    // step 2: eliminate dup
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<OffsetAndExtractBitsStmt>();
          if (bstmt_->input == stmt->input &&
              bstmt_->bit_begin == stmt->bit_begin &&
              bstmt_->bit_end == stmt->bit_end &&
              bstmt_->offset == stmt->offset) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModifiedException();
          }
        }
      }
    }
    set_done(stmt);
  }

  template <typename T>
  static bool identical_vectors(const std::vector<T> &a,
                                const std::vector<T> &b) {
    if (a.size() != b.size()) {
      return false;
    } else {
      for (int i = 0; i < (int)a.size(); i++) {
        if (a[i] != b[i])
          return false;
      }
    }
    return true;
  }

  void visit(LinearizeStmt *stmt) override {
    if (is_done(stmt))
      return;
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<LinearizeStmt>();
          if (identical_vectors(bstmt_->inputs, stmt->inputs) &&
              identical_vectors(bstmt_->strides, stmt->strides) &&
              identical_vectors(bstmt_->offsets, stmt->offsets)) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModifiedException();
          }
        }
      }
    }
    set_done(stmt);
  }

  void visit(SNodeLookupStmt *stmt) override {
    if (is_done(stmt))
      return;
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<SNodeLookupStmt>();
          if (bstmt_->snode == stmt->snode &&
              bstmt_->input_snode == stmt->input_snode &&
              bstmt_->input_index == stmt->input_index &&
              bstmt_->chid == stmt->chid &&
              // identical_vectors(bstmt_->global_indices,
              // stmt->global_indices)
              // && no need for the above line. As long as inptu_index are the
              // same global indices comparison can be omitted
              bstmt_->activate == stmt->activate) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModifiedException();
          }
        }
      }
    }
    set_done(stmt);
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

  void visit(RangeAssumptionStmt *stmt) override {
    return;
  }
};

class EliminateDup : public IRVisitor {
 public:
  StructForStmt *current_struct_for;

  EliminateDup(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    current_struct_for = nullptr;
    node->accept(this);
  }

  void visit(Block *block) override {
    std::set<int> visited;
    while (true) {
      try {
        BasicBlockEliminate _(block, visited, current_struct_for);
      } catch (IRModifiedException) {
        continue;
      }
      break;
    }
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

  void visit(StructForStmt *for_stmt) override {
    TC_ASSERT(current_struct_for == nullptr);
    current_struct_for = for_stmt;
    for_stmt->body->accept(this);
    current_struct_for = nullptr;
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
