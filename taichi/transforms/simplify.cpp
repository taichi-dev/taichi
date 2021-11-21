#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/simplify.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
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
  CompileConfig config;

  BasicBlockSimplify(Block *block,
                     std::set<int> &visited,
                     StructForStmt *current_struct_for,
                     const CompileConfig &config)
      : block(block),
        visited(visited),
        current_struct_for(current_struct_for),
        config(config) {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
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
          bool same = stmt->src == bstmt_->src;
          if (same) {
            // no store to the var?
            bool has_store = false;
            auto advanced_optimization = config.advanced_optimization;
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
                           return irpass::analysis::maybe_same_address(
                               store->dest, stmt->src);
                         else if (auto atomic = s->cast<AtomicOpStmt>())
                           return irpass::analysis::maybe_same_address(
                               atomic->dest, stmt->src);
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

  void visit(IntegerOffsetStmt *stmt) override {
    if (stmt->offset == 0) {
      stmt->replace_with(stmt->input);
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  void visit(BitExtractStmt *stmt) override {
    if (is_done(stmt))
      return;

    // step 0: eliminate empty extraction
    if (stmt->bit_begin == stmt->bit_end) {
      auto zero = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(0));
      stmt->replace_with(zero.get());
      stmt->insert_after_me(std::move(zero));
      stmt->parent->erase(current_stmt_id);
      throw IRModified();
    }

    // step 1: eliminate useless extraction of another BitExtractStmt
    if (stmt->bit_begin == 0 && stmt->input->is<BitExtractStmt>()) {
      auto bstmt = stmt->input->as<BitExtractStmt>();
      if (stmt->bit_end >= bstmt->bit_end - bstmt->bit_begin) {
        stmt->replace_with(bstmt);
        stmt->parent->erase(current_stmt_id);
        throw IRModified();
      }
    }

    // step 2: eliminate useless extraction of a LoopIndexStmt
    if (stmt->bit_begin == 0 && stmt->input->is<LoopIndexStmt>()) {
      auto bstmt = stmt->input->as<LoopIndexStmt>();
      const int max_num_bits = bstmt->max_num_bits();
      if (max_num_bits != -1 && stmt->bit_end >= max_num_bits) {
        stmt->replace_with(bstmt);
        stmt->parent->erase(current_stmt_id);
        throw IRModified();
      }
    }

    // step 3: try weakening when a struct for is used
    if (current_struct_for && !stmt->simplified) {
      const int num_loop_vars = current_struct_for->snode->num_active_indices;
      for (int k = 0; k < num_loop_vars; k++) {
        auto diff = irpass::analysis::value_diff_loop_index(
            stmt->input, current_struct_for, k);
        if (diff.linear_related() && diff.certain()) {
          // case 1: last loop var, vectorized, has assumption on vec size
          if (k == num_loop_vars - 1) {
            auto load = stmt->insert_before_me(
                Stmt::make<LoopIndexStmt>(current_struct_for, k));
            load->ret_type = PrimitiveType::i32;
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
            } else {
              if (offset != 0) {
                auto offset_const = stmt->insert_before_me(
                    Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(
                        TypedConstant(PrimitiveType::i32, offset))));
                auto sum = stmt->insert_before_me(Stmt::make<BinaryOpStmt>(
                    BinaryOpType::add, load, offset_const));
                stmt->input = sum;
              }
            }
          } else {
            // insert constant
            auto load = stmt->insert_before_me(
                Stmt::make<LoopIndexStmt>(current_struct_for, k));
            load->ret_type = PrimitiveType::i32;
            auto constant = stmt->insert_before_me(
                Stmt::make<ConstStmt>(TypedConstant(diff.low)));
            auto add = stmt->insert_before_me(
                Stmt::make<BinaryOpStmt>(BinaryOpType::add, load, constant));
            add->ret_type = PrimitiveType::i32;
            stmt->input = add;
          }
          stmt->simplified = true;
          throw IRModified();
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
    // Compare the result with 0 to make sure no overflow occurs under Debug
    // Mode.
    bool debug = config.debug;
    if (debug) {
      auto zero = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(0));
      auto check_sum =
          Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_ge, sum.get(), zero.get());
      auto assert = Stmt::make<AssertStmt>(check_sum.get(),
                                           "The indices provided are too big!",
                                           std::vector<Stmt *>());
      // Because Taichi's assertion is checked only after the execution of the
      // kernel, when the linear index overflows and goes negative, we have to
      // replace that with 0 to make sure that the rest of the kernel can still
      // complete. Otherwise, Taichi would crash due to illegal mem address.
      auto select = Stmt::make<TernaryOpStmt>(
          TernaryOpType::select, check_sum.get(), sum.get(), zero.get());

      stmt->insert_before_me(std::move(zero));
      stmt->insert_before_me(std::move(sum));
      stmt->insert_before_me(std::move(check_sum));
      stmt->insert_before_me(std::move(assert));
      stmt->replace_with(select.get());
      stmt->insert_before_me(std::move(select));
    } else {
      stmt->replace_with(sum.get());
      stmt->insert_before_me(std::move(sum));
    }
    stmt->parent->erase(stmt);
    // get types of adds and muls
    irpass::type_check(stmt->parent, config);
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
        TI_ASSERT(snode->ch[i]->dt->is_primitive(PrimitiveTypeID::i32) ||
                  snode->ch[i]->dt->is_primitive(PrimitiveTypeID::f32));
      }

      auto offset_stmt = stmt->insert_after_me(Stmt::make<IntegerOffsetStmt>(
          stmt, previous_offset->offset * sizeof(int32) * (snode->ch.size())));

      stmt->input_index = previous_offset->input;
      stmt->replace_with(offset_stmt);
      offset_stmt->as<IntegerOffsetStmt>()->input = stmt;
      throw IRModified();
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

    set_done(stmt);
  }

  void visit(WhileControlStmt *stmt) override {
    if (stmt->width() == 1 && stmt->mask) {
      stmt->mask = nullptr;
      throw IRModified();
    }
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
              lanes.push_back(LocalAddress(store->dest, l));
            }
            auto load =
                if_stmt->insert_before_me(Stmt::make<LocalLoadStmt>(lanes));
            irpass::type_check(load, config);
            auto select = if_stmt->insert_before_me(
                Stmt::make<TernaryOpStmt>(TernaryOpType::select, if_stmt->cond,
                                          true_branch ? store->val : load,
                                          true_branch ? load : store->val));
            irpass::type_check(select, config);
            store->val = select;
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

    if (config.flatten_if) {
      if (if_stmt->true_statements &&
          flatten(if_stmt->true_statements->statements, true)) {
        throw IRModified();
      }
      if (if_stmt->false_statements &&
          flatten(if_stmt->false_statements->statements, false)) {
        throw IRModified();
      }
    }

    if (if_stmt->true_statements) {
      if (if_stmt->true_statements->statements.empty()) {
        if_stmt->set_true_statements(nullptr);
        throw IRModified();
      }
    }

    if (if_stmt->false_statements) {
      if (if_stmt->false_statements->statements.empty()) {
        if_stmt->set_false_statements(nullptr);
        throw IRModified();
      }
    }

    if (!if_stmt->true_statements && !if_stmt->false_statements) {
      if_stmt->parent->erase(if_stmt);
      throw IRModified();
    }

    if (config.advanced_optimization) {
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
  const CompileConfig &config;

  Simplify(IRNode *node, const CompileConfig &config) : config(config) {
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
        BasicBlockSimplify _(block, visited, current_struct_for, config);
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
    TI_ASSERT_INFO(current_struct_for == nullptr,
                   "Nested struct-fors are not supported for now. "
                   "Please try to use range-fors for inner loops.");
    current_struct_for = for_stmt;
    for_stmt->body->accept(this);
    current_struct_for = nullptr;
  }

  void visit(MeshForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) override {
    stmt->all_blocks_accept(this);
  }
};

const PassID FullSimplifyPass::id = "FullSimplifyPass";

namespace irpass {

bool simplify(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  bool modified = false;
  while (true) {
    Simplify pass(root, config);
    if (pass.modified)
      modified = true;
    else
      break;
  }
  return modified;
}

void full_simplify(IRNode *root,
                   const CompileConfig &config,
                   const FullSimplifyPass::Args &args) {
  TI_AUTO_PROF;
  if (config.advanced_optimization) {
    bool first_iteration = true;
    while (true) {
      bool modified = false;
      if (extract_constant(root, config))
        modified = true;
      if (unreachable_code_elimination(root))
        modified = true;
      if (binary_op_simplify(root, config))
        modified = true;
      if (config.constant_folding &&
          constant_fold(root, config, {args.program}))
        modified = true;
      if (die(root))
        modified = true;
      if (alg_simp(root, config))
        modified = true;
      if (loop_invariant_code_motion(root, config))
        modified = true;
      if (die(root))
        modified = true;
      if (simplify(root, config))
        modified = true;
      if (die(root))
        modified = true;
      if (whole_kernel_cse(root))
        modified = true;
      // Don't do this time-consuming optimization pass again if the IR is
      // not modified.
      if ((first_iteration || modified) && config.cfg_optimization &&
          cfg_optimization(root, args.after_lower_access))
        modified = true;
      first_iteration = false;
      if (!modified)
        break;
    }
    return;
  }
  if (config.constant_folding) {
    constant_fold(root, config, {args.program});
    die(root);
  }
  simplify(root, config);
  die(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
