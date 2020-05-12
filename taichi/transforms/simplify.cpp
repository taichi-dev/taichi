#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include <set>
#include <unordered_set>
#include <utility>

TLANG_NAMESPACE_BEGIN

// Common subexpression elimination, store forwarding, useless local store
// elimination; Simplify if statements into conditional stores.
class BasicBlockSimplify : public IRVisitor {
 public:
  Block *block;

  int current_stmt_id;
  std::set<int> &visited;
  StructForStmt *current_struct_for;
  Kernel *kernel;

  BasicBlockSimplify(Block *block,
                     std::set<int> &visited,
                     StructForStmt *current_struct_for,
                     Kernel *kernel)
      : block(block),
        visited(visited),
        current_struct_for(current_struct_for),
        kernel(kernel) {
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
      TI_ERROR("Visitor for non-container stmt undefined.");
    }
  }

  void visit(GlobalPtrStmt *stmt) override {
    if (is_done(stmt))
      return;
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      // A previous GlobalPtrStmt in the basic block
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
            if (bstmt_->activate == stmt->activate || bstmt_->activate) {
              stmt->replace_with(bstmt.get());
              stmt->parent->erase(current_stmt_id);
              throw IRModified();
            } else {
              // bstmt->active == false && stmt->active == true
              // do nothing.
            }
          }
        }
      }
    }
    set_done(stmt);
  }

  void visit(ArgLoadStmt *stmt) override {
    if (is_done(stmt))
      return;
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      auto &bstmt_data = *bstmt;
      if (typeid(bstmt_data) == typeid(*stmt)) {
        if (stmt->width() == bstmt->width()) {
          auto bstmt_ = bstmt->as<ArgLoadStmt>();
          bool same = stmt->arg_id == bstmt_->arg_id;
          if (same) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModified();
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
            throw IRModified();
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
          stmt->width() == stmt->elements[0].stmt->width()) {
        // useless shuffle.
        stmt->replace_with(stmt->elements[0].stmt);
        stmt->parent->erase(current_stmt_id);
        throw IRModified();
      }
    }

    // weaken indexing
    if (current_struct_for && stmt->width() == 1) {
      auto &loop_vars = current_struct_for->loop_vars;
      for (int k = 0; k < (int)loop_vars.size(); k++) {
        auto diff = irpass::analysis::value_diff(
            stmt->elements[0].stmt, stmt->elements[0].index,
            current_struct_for->loop_vars[k]);
        if (diff.linear_related() && diff.certain()) {
          auto load = stmt->insert_before_me(
              Stmt::make<LocalLoadStmt>(LocalAddress(loop_vars[k], 0)));
          load->ret_type.data_type = DataType::i32;
          auto constant = stmt->insert_before_me(
              Stmt::make<ConstStmt>(TypedConstant(diff.low)));
          constant->ret_type.data_type = DataType::i32;
          auto add = stmt->insert_before_me(
              Stmt::make<BinaryOpStmt>(BinaryOpType::add, load, constant));
          add->ret_type.data_type = DataType::i32;
          stmt->replace_with(add);
          stmt->parent->erase(stmt);
          throw IRModified();
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
            throw IRModified();
          }
        }
      }
    }
    set_done(stmt);
  }

  // Local variable operation optimizations:
  // 1. Store forwarding
  //

  bool modifies_local(Stmt *stmt, std::vector<Stmt *> vars) {
    if (stmt->is<LocalStoreStmt>()) {
      auto st = stmt->as<LocalStoreStmt>();
      for (auto var : vars) {
        if (st->ptr == var) {
          return true;
        }
      }
    } else if (stmt->is<AtomicOpStmt>()) {
      auto st = stmt->as<AtomicOpStmt>();
      for (auto var : vars) {
        if (st->dest == var) {
          return true;
        }
      }
    }
    return false;
  }

  void visit(LocalLoadStmt *stmt) override {
    if (is_done(stmt))
      return;

    // Merge identical loads
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
              if (!advanced_optimization) {
                if (block->statements[j]
                        ->is_container_statement()) {  // no if, while, etc..
                  has_related_store = true;
                  break;
                }
                if (modifies_local(block->statements[j].get(), vars)) {
                  has_related_store = true;
                }
                continue;
              }
              if (irpass::analysis::has_store_or_atomic(
                      block->statements[j].get(), vars)) {
                has_related_store = true;
                break;
              }
            }
            if (!has_related_store) {
              stmt->replace_with(bstmt.get());
              stmt->parent->erase(current_stmt_id);
              throw IRModified();
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
      // Check all previous statements in the current block before the local
      // load
      Stmt *containing_statement = stmt;
      auto stmt_id = block->locate(containing_statement);
      TI_ASSERT(stmt_id != -1);
      for (int i = stmt_id - 1; i >= 0; i--) {
        if (!advanced_optimization) {
          auto &bstmt = block->statements[i];
          // Find a previous store
          if (auto s = bstmt->cast<AtomicOpStmt>()) {
            if (s->dest == alloca) {
              break;
            }
          }
          if (bstmt->is<LocalStoreStmt>()) {
            auto bstmt_ = bstmt->as<LocalStoreStmt>();
            // Same alloca
            if (bstmt_->ptr == alloca) {
              // Forward to the first local store only
              stmt->replace_with(bstmt_->data);
              stmt->parent->erase(current_stmt_id);
              throw IRModified();
            }
          } else if (bstmt->is_container_statement()) {
            // assume this container may modify the local var
            break;
          }
          continue;
        }
        auto last_store = irpass::analysis::last_store_or_atomic(
            block->statements[i].get(), alloca);
        if (!last_store.first) {
          // invalid
          break;
        }
        auto bstmt = last_store.second;
        if (bstmt != nullptr) {
          if (bstmt->is<LocalStoreStmt>()) {
            // Forward to the first local store only
            stmt->replace_with(bstmt->as<LocalStoreStmt>()->data);
            stmt->parent->erase(current_stmt_id);
            throw IRModified();
          } else {
            TI_ASSERT(bstmt->is<AllocaStmt>());
            auto zero = stmt->insert_after_me(Stmt::make<ConstStmt>(
                LaneAttribute<TypedConstant>(bstmt->ret_type.data_type)));
            zero->repeat(stmt->width());
            stmt->replace_with(zero);
            stmt->parent->erase(current_stmt_id);
            throw IRModified();
          }
        }
      }
      // Note: simply checking all statements before stmt is not sufficient
      // since statements after stmt may change the value of the alloca
      // For example, in a loop, later part of the loop body may alter the local
      // var value.
    }
    set_done(stmt);
  }

  void visit(LocalStoreStmt *stmt) override {
    if (is_done(stmt))
      return;

    // has previous store?
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<LocalStoreStmt>();
          bool same = stmt->ptr == bstmt_->ptr;
          if (same) {
            bool has_load = false;
            for (int j = i + 1; j < current_stmt_id; j++) {
              if (!advanced_optimization) {
                if (block->statements[j]
                        ->is_container_statement()) {  // no if, while, etc..
                  has_load = true;
                  break;
                }
                if (block->statements[j]->is<LocalLoadStmt>() &&
                    block->statements[j]->as<LocalLoadStmt>()->has_source(
                        stmt->ptr)) {
                  has_load = true;
                }
                if (block->statements[j]->is<AtomicOpStmt>() &&
                    (block->statements[j]->as<AtomicOpStmt>()->dest ==
                     stmt->ptr)) {
                  // $a = alloca
                  // $b : local store [$a <- v1]  <-- prev lstore |bstmt_|
                  // $c = atomic add($a, v2)      <-- cannot eliminate $b
                  // $d : local store [$a <- v3]
                  has_load = true;
                }
                continue;
              }
              if (!irpass::analysis::gather_statements(
                       block->statements[j].get(),
                       [&](Stmt *s) {
                         if (auto load = s->cast<LocalLoadStmt>())
                           return load->has_source(stmt->ptr);
                         else if (auto atomic = s->cast<AtomicOpStmt>())
                           return atomic->dest == stmt->ptr;
                         else
                           return s->is<ContinueStmt>() ||
                                  s->is<WhileControlStmt>();
                       })
                       .empty()) {
                has_load = true;
                break;
              }
            }
            if (!has_load) {
              stmt->parent->erase(bstmt_);
              throw IRModified();
            }
          }
        }
      }
    }

    // Does it have a following load? If not, delete.
    if (stmt->parent->locate(stmt->ptr) != -1) {
      // optimize variables local to this block only
      bool has_related = false;
      for (int i = current_stmt_id + 1; i < (int)block->statements.size();
           i++) {
        if (!advanced_optimization) {
          auto &bstmt = block->statements[i];
          if (bstmt->is_container_statement()) {
            has_related = true;
            break;
          }
          if (bstmt->is<LocalLoadStmt>()) {
            auto bstmt_ = bstmt->as<LocalLoadStmt>();
            if (bstmt_->has_source(stmt->ptr)) {
              has_related = true;
              break;
            }
          }
          if (bstmt->is<AtomicOpStmt>()) {
            // $a = alloca
            // $b : local store [$a <- v1]
            // $c = atomic add($a, v2)      <-- cannot eliminate $b
            auto bstmt_ = bstmt->as<AtomicOpStmt>();
            if (bstmt_->dest == stmt->ptr) {
              has_related = true;
              break;
            }
          }
          continue;
        }
        if (!irpass::analysis::gather_statements(
                 block->statements[i].get(),
                 [&](Stmt *s) {
                   if (auto load = s->cast<LocalLoadStmt>())
                     return load->has_source(stmt->ptr);
                   else if (auto atomic = s->cast<AtomicOpStmt>())
                     return atomic->dest == stmt->ptr;
                   else
                     return false;
                 })
                 .empty()) {
          has_related = true;
          break;
        }
      }
      if (!has_related) {
        stmt->parent->erase(stmt);
        throw IRModified();
      }
    }

    set_done(stmt);
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
              if (!advanced_optimization) {
                if (block->statements[j]
                        ->is_container_statement()) {  // no if, while, etc..
                  has_store = true;
                  break;
                }
                if (block->statements[j]->is<GlobalStoreStmt>()) {
                  has_store = true;
                }
                continue;
              }
              if (!irpass::analysis::gather_statements(
                       block->statements[j].get(),
                       [&](Stmt *s) {
                         if (auto store = s->cast<GlobalStoreStmt>())
                           return maybe_same_address(store->ptr, stmt->ptr);
                         else if (auto atomic = s->cast<AtomicOpStmt>())
                           return maybe_same_address(atomic->dest, stmt->ptr);
                         else
                           return false;
                       })
                       .empty()) {
                has_store = true;
                break;
              }
            }
            if (!has_store) {
              stmt->replace_with(bstmt.get());
              stmt->parent->erase(current_stmt_id);
              throw IRModified();
            }
          }
        }
      }
    }
    set_done(stmt);
  }

  void visit(GlobalStoreStmt *stmt) override {
    // do nothing for global mem ops
  }

  void visit(SNodeOpStmt *stmt) override {
    return;
  }

  void visit(IntegerOffsetStmt *stmt) override {
    if (is_done(stmt))
      return;
    if (stmt->offset == 0) {
      stmt->replace_with(stmt->input);
      stmt->parent->erase(stmt);
      throw IRModified();
    }

    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<IntegerOffsetStmt>();
          if (bstmt_->input == stmt->input && bstmt_->offset == stmt->offset) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModified();
          }
        }
      }
    }
    set_done(stmt);
  }

  void visit(GetRootStmt *stmt) override {
    if (is_done(stmt))
      return;
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (bstmt->is<GetRootStmt>()) {
        stmt->replace_with(bstmt.get());
        stmt->parent->erase(current_stmt_id);
        throw IRModified();
      }
    }
    set_done(stmt);
  }

  void visit(UnaryOpStmt *stmt) override {
    if (is_done(stmt))
      return;
    if (stmt->is_cast()) {
      if (stmt->cast_type == stmt->operand->ret_type.data_type) {
        stmt->replace_with(stmt->operand);
        stmt->parent->erase(current_stmt_id);
        set_done(stmt);
        return;
      }
    }
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<UnaryOpStmt>();
          if (bstmt_->same_operation(stmt) &&
              bstmt_->operand == stmt->operand) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModified();
          }
        }
      }
    }
    set_done(stmt);
  }

  void visit(BinaryOpStmt *stmt) override {
    if (is_done(stmt))
      return;
    if ((stmt->op_type == BinaryOpType::add ||
         stmt->op_type == BinaryOpType::sub) &&
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
          throw IRModified();
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
            throw IRModified();
          }
        }
      }
    }
    set_done(stmt);
  }

  void visit(TernaryOpStmt *stmt) override {
    if (is_done(stmt))
      return;
    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<TernaryOpStmt>();
          if (bstmt_->op_type == stmt->op_type && bstmt_->op1 == stmt->op1 &&
              bstmt_->op2 == stmt->op2 && bstmt_->op3 == stmt->op3) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModified();
          }
        }
      }
    }
    set_done(stmt);
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override {
    if (is_done(stmt))
      return;
    // step 1: try weakening when a struct for is used
    if (current_struct_for && !stmt->simplified) {
      auto &loop_vars = current_struct_for->loop_vars;
      for (int k = 0; k < (int)loop_vars.size(); k++) {
        auto diff = irpass::analysis::value_diff(
            stmt->input, 0, current_struct_for->loop_vars[k]);
        if (diff.linear_related() && diff.certain()) {
          // case 1: last loop var, vectorized, has assumption on vec size
          if (k == (int)current_struct_for->loop_vars.size() - 1) {
            auto load = stmt->insert_before_me(
                Stmt::make<LocalLoadStmt>(LocalAddress(loop_vars[k], 0)));
            load->ret_type.data_type = DataType::i32;
            stmt->input = load;
            int64 bound = 1LL << stmt->bit_end;
            auto offset = (((int64)diff.low % bound + bound) % bound) &
                          ~((1LL << (stmt->bit_begin)) - 1);

            if (current_struct_for->vectorize == 1)
              offset = diff.low;
            if (stmt->bit_begin == 0 &&
                current_struct_for->vectorize == bound) {
              // TODO: take care of cases where vectorization width != z
              // dimension of the block
              auto offset_stmt = stmt->insert_after_me(
                  Stmt::make<IntegerOffsetStmt>(stmt, offset));
              stmt->replace_with(offset_stmt);
              // fix the offset stmt operand
              offset_stmt->as<IntegerOffsetStmt>()->input = stmt;
              stmt->offset = 0;
            } else {
              stmt->offset = offset;
            }
          } else {
            // insert constant
            auto load = stmt->insert_before_me(
                Stmt::make<LocalLoadStmt>(LocalAddress(loop_vars[k], 0)));
            load->ret_type.data_type = DataType::i32;
            auto constant = stmt->insert_before_me(
                Stmt::make<ConstStmt>(TypedConstant(diff.low)));
            auto add = stmt->insert_before_me(
                Stmt::make<BinaryOpStmt>(BinaryOpType::add, load, constant));
            add->ret_type.data_type = DataType::i32;
            stmt->input = add;
          }
          stmt->simplified = true;
          throw IRModified();
        }
      }
    }

    // step 2: eliminate useless extraction of another OffsetAndExtractBitsStmt
    if (advanced_optimization) {
      if (stmt->offset == 0 && stmt->bit_begin == 0 &&
          stmt->input->is<OffsetAndExtractBitsStmt>()) {
        auto bstmt = stmt->input->as<OffsetAndExtractBitsStmt>();
        if (stmt->bit_end == bstmt->bit_end - bstmt->bit_begin) {
          stmt->replace_with(bstmt);
          stmt->parent->erase(current_stmt_id);
          throw IRModified();
        }
      }
    }

    // step 3: eliminate dup
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
            throw IRModified();
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
    if (!stmt->inputs.empty() && stmt->inputs.back()->is<IntegerOffsetStmt>()) {
      auto previous_offset = stmt->inputs.back()->as<IntegerOffsetStmt>();
      // push forward offset
      auto offset_stmt = stmt->insert_after_me(
          Stmt::make<IntegerOffsetStmt>(stmt, previous_offset->offset));

      stmt->inputs.back() = previous_offset->input;
      stmt->replace_with(offset_stmt);
      offset_stmt->as<IntegerOffsetStmt>()->input = stmt;
      throw IRModified();
    }

    // Lower into a series of adds and muls.
    auto sum = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(0));
    auto stride_product = 1;
    for (int i = (int)stmt->inputs.size() - 1; i >= 0; i--) {
      auto stride_stmt =
          Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(stride_product));
      auto mul = Stmt::make<BinaryOpStmt>(BinaryOpType::mul, stmt->inputs[i],
                                          stride_stmt.get());
      auto newsum =
          Stmt::make<BinaryOpStmt>(BinaryOpType::add, sum.get(), mul.get());
      stmt->insert_before_me(std::move(sum));
      sum = std::move(newsum);
      stmt->insert_before_me(std::move(stride_stmt));
      stmt->insert_before_me(std::move(mul));
      stride_product *= stmt->strides[i];
    }
    stmt->replace_with(sum.get());
    stmt->insert_before_me(std::move(sum));
    stmt->parent->erase(stmt);
    // get types of adds and muls
    irpass::typecheck(stmt->parent, kernel);
    throw IRModified();
  }

  void visit(SNodeLookupStmt *stmt) override {
    if (is_done(stmt))
      return;

    if (stmt->input_index->is<IntegerOffsetStmt>()) {
      auto previous_offset = stmt->input_index->as<IntegerOffsetStmt>();
      // push forward offset

      auto snode = stmt->snode;
      // compute offset...
      for (int i = 0; i < (int)snode->ch.size(); i++) {
        TI_ASSERT(snode->ch[i]->type == SNodeType::place);
        TI_ASSERT(snode->ch[i]->dt == DataType::i32 ||
                  snode->ch[i]->dt == DataType::f32);
      }

      auto offset_stmt = stmt->insert_after_me(Stmt::make<IntegerOffsetStmt>(
          stmt, previous_offset->offset * sizeof(int32) * (snode->ch.size())));

      stmt->input_index = previous_offset->input;
      stmt->replace_with(offset_stmt);
      offset_stmt->as<IntegerOffsetStmt>()->input = stmt;
      throw IRModified();
    }

    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<SNodeLookupStmt>();
          if (bstmt_->snode == stmt->snode &&
              bstmt_->input_snode == stmt->input_snode &&
              bstmt_->input_index == stmt->input_index &&
              // identical_vectors(bstmt_->global_indices,
              // stmt->global_indices)
              // && no need for the above line. As long as input_index are the
              // same global indices comparison can be omitted
              bstmt_->activate == stmt->activate) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModified();
          }
        }
      }
    }
    set_done(stmt);
  }

  void visit(GetChStmt *stmt) override {
    if (is_done(stmt))
      return;

    if (stmt->input_ptr->is<IntegerOffsetStmt>()) {
      auto previous_offset = stmt->input_ptr->as<IntegerOffsetStmt>();
      // push forward offset

      // auto snode = stmt->input_snode;
      auto offset_stmt = stmt->insert_after_me(Stmt::make<IntegerOffsetStmt>(
          stmt, stmt->chid * sizeof(int32) + previous_offset->offset));

      stmt->input_ptr = previous_offset->input;
      stmt->replace_with(offset_stmt);
      stmt->chid = 0;
      stmt->output_snode = stmt->input_snode->ch[stmt->chid].get();
      offset_stmt->as<IntegerOffsetStmt>()->input = stmt;
      throw IRModified();
    }

    for (int i = 0; i < current_stmt_id; i++) {
      auto &bstmt = block->statements[i];
      if (stmt->ret_type == bstmt->ret_type) {
        auto &bstmt_data = *bstmt;
        if (typeid(bstmt_data) == typeid(*stmt)) {
          auto bstmt_ = bstmt->as<GetChStmt>();
          if (bstmt_->input_ptr == stmt->input_ptr &&
              bstmt_->input_snode == stmt->input_snode &&
              bstmt_->chid == stmt->chid) {
            stmt->replace_with(bstmt.get());
            stmt->parent->erase(current_stmt_id);
            throw IRModified();
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
    if (stmt->width() == 1 && stmt->mask) {
      stmt->mask = nullptr;
      throw IRModified();
    }
  }

  void visit(ContinueStmt *stmt) override {
    return;
  }

  static bool is_global_write(Stmt *stmt) {
    return stmt->is<GlobalStoreStmt>() || stmt->is<AtomicOpStmt>();
  }

  static bool is_atomic_value_used(const std::vector<pStmt> &clause,
                                   int atomic_stmt_i) {
    // Cast type to check precondition
    const auto *stmt = clause[atomic_stmt_i]->as<AtomicOpStmt>();
    auto alloca = stmt->dest;

    for (std::size_t i = atomic_stmt_i + 1; i < clause.size(); ++i) {
      for (const auto &op : clause[i]->get_operands()) {
        if (op && (op->instance_id == stmt->instance_id ||
                   op->instance_id == alloca->instance_id)) {
          return true;
        }
      }
    }
    return false;
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->width() == 1 && (if_stmt->true_mask || if_stmt->false_mask)) {
      if_stmt->true_mask = nullptr;
      if_stmt->false_mask = nullptr;
      throw IRModified();
    }
    auto flatten = [&](std::vector<pStmt> &clause, bool true_branch) {
      bool plain_clause = true;  // no global store, no container

      // Here we try to move statements outside the clause;
      // Keep only global atomics/store, and other statements that have no
      // global side effects. LocalStore is kept and specially treated later.

      bool global_state_changed = false;
      for (int i = 0; i < (int)clause.size() && plain_clause; i++) {
        bool has_side_effects = clause[i]->is_container_statement() ||
                                clause[i]->has_global_side_effect();

        if (global_state_changed && clause[i]->is<GlobalLoadStmt>()) {
          // This clause cannot be trivially simplified, since there's a global
          // load after store and they must be kept in order
          plain_clause = false;
        }

        if (clause[i]->is<GlobalStoreStmt>() ||
            clause[i]->is<LocalStoreStmt>() || !has_side_effects) {
          // This stmt can be kept.
        } else if (clause[i]->is<AtomicOpStmt>()) {
          plain_clause = plain_clause && !is_atomic_value_used(clause, i);
        } else {
          plain_clause = false;
        }
        if (is_global_write(clause[i].get()) || has_side_effects) {
          global_state_changed = true;
        }
      }
      if (plain_clause) {
        for (int i = 0; i < (int)clause.size(); i++) {
          if (is_global_write(clause[i].get())) {
            // do nothing. Keep the statement.
            continue;
          }
          if (clause[i]->is<LocalStoreStmt>()) {
            auto store = clause[i]->as<LocalStoreStmt>();
            auto lanes = LaneAttribute<LocalAddress>();
            for (int l = 0; l < store->width(); l++) {
              lanes.push_back(LocalAddress(store->ptr, l));
            }
            auto load =
                if_stmt->insert_before_me(Stmt::make<LocalLoadStmt>(lanes));
            load->infer_type();
            auto select = if_stmt->insert_before_me(
                Stmt::make<TernaryOpStmt>(TernaryOpType::select, if_stmt->cond,
                                          true_branch ? store->data : load,
                                          true_branch ? load : store->data));
            select->infer_type();
            store->data = select;
            if_stmt->insert_before_me(std::move(clause[i]));
          } else {
            if_stmt->insert_before_me(std::move(clause[i]));
          }
        }
        auto clean_clause = std::vector<pStmt>();
        bool reduced = false;
        for (auto &&stmt : clause) {
          if (stmt != nullptr) {
            clean_clause.push_back(std::move(stmt));
          } else {
            reduced = true;
          }
        }
        clause = std::move(clean_clause);
        return reduced;
      }
      return false;
    };

    if (if_stmt->true_statements &&
        flatten(if_stmt->true_statements->statements, true)) {
      throw IRModified();
    }
    if (if_stmt->false_statements &&
        flatten(if_stmt->false_statements->statements, false)) {
      throw IRModified();
    }

    if (if_stmt->true_statements) {
      if (if_stmt->true_statements->statements.empty()) {
        if_stmt->true_statements = nullptr;
        throw IRModified();
      }
    }

    if (if_stmt->false_statements) {
      if (if_stmt->false_statements->statements.empty()) {
        if_stmt->false_statements = nullptr;
        throw IRModified();
      }
    }

    if (!if_stmt->true_statements && !if_stmt->false_statements) {
      if_stmt->parent->erase(if_stmt);
      throw IRModified();
    }

    if (advanced_optimization) {
      // Merge adjacent if's with the identical condition.
      // TODO: What about IfStmt::true_mask and IfStmt::false_mask?
      if (current_stmt_id > 0 &&
          block->statements[current_stmt_id - 1]->is<IfStmt>()) {
        auto bstmt = block->statements[current_stmt_id - 1]->as<IfStmt>();
        if (bstmt->cond == if_stmt->cond) {
          auto concatenate = [](std::unique_ptr<Block> &clause1,
                                std::unique_ptr<Block> &clause2) {
            if (clause1 == nullptr) {
              clause1 = std::move(clause2);
              return;
            }
            if (clause2 != nullptr)
              clause1->insert(VecStatement(std::move(clause2->statements)));
          };
          concatenate(bstmt->true_statements, if_stmt->true_statements);
          concatenate(bstmt->false_statements, if_stmt->false_statements);
          if_stmt->parent->erase(if_stmt);
          throw IRModified();
        }
      }
    }
  }

  void visit(RangeAssumptionStmt *stmt) override {
    return;
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->has_body() && stmt->body->statements.empty()) {
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  void visit(WhileStmt *stmt) override {
    if (stmt->width() == 1 && stmt->mask) {
      stmt->mask = nullptr;
      throw IRModified();
    }
  }
};

class Simplify : public IRVisitor {
 public:
  StructForStmt *current_struct_for;
  bool modified;
  Kernel *kernel;

  Simplify(IRNode *node, Kernel *kernel) : kernel(kernel) {
    modified = false;
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    current_struct_for = nullptr;
    node->accept(this);
  }

  void visit(Block *block) override {
    std::set<int> visited;
    while (true) {
      try {
        BasicBlockSimplify _(block, visited, current_struct_for, kernel);
      } catch (IRModified) {
        modified = true;
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
    if (if_stmt->false_statements)
      if_stmt->false_statements->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    TI_ASSERT(current_struct_for == nullptr);
    current_struct_for = for_stmt;
    for_stmt->body->accept(this);
    current_struct_for = nullptr;
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->body)
      stmt->body->accept(this);
  }
};

namespace irpass {

void simplify(IRNode *root, Kernel *kernel) {
  while (1) {
    Simplify pass(root, kernel);
    if (!pass.modified)
      break;
  }
}

void full_simplify(IRNode *root, const CompileConfig &config, Kernel *kernel) {
  constant_fold(root);
  if (advanced_optimization) {
    alg_simp(root, config);
    die(root);
    whole_kernel_cse(root);
  }
  die(root);
  simplify(root, kernel);
  die(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
