#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/simplify.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/transforms/utils.h"
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
  DelayedIRModifier modifier;

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
  }

  bool is_done(Stmt *stmt) {
    return visited.find(stmt->instance_id) != visited.end();
  }

  void set_done(Stmt *stmt) {
    visited.insert(stmt->instance_id);
  }

  void accept_block() {
    for (int i = 0; i < (int)block->statements.size(); i++) {
      current_stmt_id = i;
      block->statements[i]->accept(this);
    }
  }

  static bool run(Block *block,
                  std::set<int> &visited,
                  StructForStmt *current_struct_for,
                  const CompileConfig &config) {
    BasicBlockSimplify simplifier(block, visited, current_struct_for, config);
    bool ir_modified = false;
    while (true) {
      simplifier.accept_block();
      if (simplifier.modifier.modify_ir()) {
        ir_modified = true;
      } else {
        break;
      }
    }
    return ir_modified;
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
              if (block->statements[j]->is<FuncCallStmt>()) {
                has_store = true;
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
              stmt->replace_usages_with(bstmt.get());
              modifier.erase(stmt);
              return;
            }
          }
        }
      }
    }
    set_done(stmt);
  }

  void visit(IntegerOffsetStmt *stmt) override {
    if (stmt->offset == 0) {
      stmt->replace_usages_with(stmt->input);
      modifier.erase(stmt);
    }
  }

  void visit(BitExtractStmt *stmt) override {
    if (is_done(stmt))
      return;

    // step 0: eliminate empty extraction
    if (stmt->bit_begin == stmt->bit_end) {
      auto zero = Stmt::make<ConstStmt>(TypedConstant(0));
      stmt->replace_usages_with(zero.get());
      modifier.insert_after(stmt, std::move(zero));
      modifier.erase(stmt);
      return;
    }

    // step 1: eliminate useless extraction of another BitExtractStmt
    if (stmt->bit_begin == 0 && stmt->input->is<BitExtractStmt>()) {
      auto bstmt = stmt->input->as<BitExtractStmt>();
      if (stmt->bit_end >= bstmt->bit_end - bstmt->bit_begin) {
        stmt->replace_usages_with(bstmt);
        modifier.erase(stmt);
        return;
      }
    }

    // step 2: eliminate useless extraction of a LoopIndexStmt
    if (stmt->bit_begin == 0 && stmt->input->is<LoopIndexStmt>()) {
      auto bstmt = stmt->input->as<LoopIndexStmt>();
      const int max_num_bits = bstmt->max_num_bits();
      if (max_num_bits != -1 && stmt->bit_end >= max_num_bits) {
        stmt->replace_usages_with(bstmt);
        modifier.erase(stmt);
        return;
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
            auto load = Stmt::make<LoopIndexStmt>(current_struct_for, k);
            load->ret_type = PrimitiveType::i32;
            stmt->input = load.get();
            int64 bound = 1LL << stmt->bit_end;
            auto offset = (((int64)diff.low % bound + bound) % bound) &
                          ~((1LL << (stmt->bit_begin)) - 1);
            auto load_addr = load.get();
            modifier.insert_before(stmt, std::move(load));
            offset = diff.low;                         // TODO: Vectorization
            if (stmt->bit_begin == 0 && bound == 1) {  // TODO: Vectorization
              // TODO: take care of cases where vectorization width != z
              // dimension of the block
              auto offset_stmt = Stmt::make<IntegerOffsetStmt>(stmt, offset);
              stmt->replace_usages_with(offset_stmt.get());
              // fix the offset stmt operand
              offset_stmt->as<IntegerOffsetStmt>()->input = stmt;
              modifier.insert_after(stmt, std::move(offset_stmt));
            } else {
              if (offset != 0) {
                auto offset_const = Stmt::make<ConstStmt>(
                    TypedConstant(PrimitiveType::i32, offset));
                auto sum = Stmt::make<BinaryOpStmt>(
                    BinaryOpType::add, load_addr, offset_const.get());
                stmt->input = sum.get();
                modifier.insert_before(stmt, std::move(offset_const));
                modifier.insert_before(stmt, std::move(offset_const));
              }
            }
          } else {
            // insert constant
            auto load = Stmt::make<LoopIndexStmt>(current_struct_for, k);
            load->ret_type = PrimitiveType::i32;
            auto constant = Stmt::make<ConstStmt>(TypedConstant(diff.low));
            auto add = Stmt::make<BinaryOpStmt>(BinaryOpType::add, load.get(),
                                                constant.get());
            add->ret_type = PrimitiveType::i32;
            stmt->input = add.get();
            modifier.insert_before(stmt, std::move(load));
            modifier.insert_before(stmt, std::move(constant));
            modifier.insert_before(stmt, std::move(add));
          }
          stmt->simplified = true;
          return;
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
      auto offset_stmt =
          Stmt::make<IntegerOffsetStmt>(stmt, previous_offset->offset);

      stmt->inputs.back() = previous_offset->input;
      stmt->replace_usages_with(offset_stmt.get());
      offset_stmt->as<IntegerOffsetStmt>()->input = stmt;
      modifier.insert_after(stmt, std::move(offset_stmt));
      return;
    }

    // Lower into a series of adds and muls.
    auto sum = Stmt::make<ConstStmt>(TypedConstant(0));
    auto stride_product = 1;
    for (int i = (int)stmt->inputs.size() - 1; i >= 0; i--) {
      auto stride_stmt = Stmt::make<ConstStmt>(TypedConstant(stride_product));
      auto mul = Stmt::make<BinaryOpStmt>(BinaryOpType::mul, stmt->inputs[i],
                                          stride_stmt.get());
      auto newsum =
          Stmt::make<BinaryOpStmt>(BinaryOpType::add, sum.get(), mul.get());
      modifier.insert_before(stmt, std::move(sum));
      sum = std::move(newsum);
      modifier.insert_before(stmt, std::move(stride_stmt));
      modifier.insert_before(stmt, std::move(mul));
      stride_product *= stmt->strides[i];
    }
    // Compare the result with 0 to make sure no overflow occurs under Debug
    // Mode.
    bool debug = config.debug;
    if (debug) {
      auto zero = Stmt::make<ConstStmt>(TypedConstant(0));
      auto check_sum =
          Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_ge, sum.get(), zero.get());
      auto assert = Stmt::make<AssertStmt>(
          check_sum.get(), "The indices provided are too big!\n" + stmt->tb,
          std::vector<Stmt *>());
      // Because Taichi's assertion is checked only after the execution of the
      // kernel, when the linear index overflows and goes negative, we have to
      // replace that with 0 to make sure that the rest of the kernel can still
      // complete. Otherwise, Taichi would crash due to illegal mem address.
      auto select = Stmt::make<TernaryOpStmt>(
          TernaryOpType::select, check_sum.get(), sum.get(), zero.get());

      modifier.insert_before(stmt, std::move(zero));
      modifier.insert_before(stmt, std::move(sum));
      modifier.insert_before(stmt, std::move(check_sum));
      modifier.insert_before(stmt, std::move(assert));
      stmt->replace_usages_with(select.get());
      modifier.insert_before(stmt, std::move(select));
    } else {
      stmt->replace_usages_with(sum.get());
      modifier.insert_before(stmt, std::move(sum));
    }
    modifier.erase(stmt);
    // get types of adds and muls
    modifier.type_check(stmt->parent, config);
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

      auto offset_stmt = Stmt::make<IntegerOffsetStmt>(
          stmt, previous_offset->offset * sizeof(int32) * (snode->ch.size()));

      stmt->input_index = previous_offset->input;
      stmt->replace_usages_with(offset_stmt.get());
      offset_stmt->as<IntegerOffsetStmt>()->input = stmt;
      modifier.insert_after(stmt, std::move(offset_stmt));
      return;
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
      auto offset_stmt = Stmt::make<IntegerOffsetStmt>(
          stmt, stmt->chid * sizeof(int32) + previous_offset->offset);

      stmt->input_ptr = previous_offset->input;
      stmt->replace_usages_with(offset_stmt.get());
      stmt->chid = 0;
      stmt->output_snode = stmt->input_snode->ch[stmt->chid].get();
      offset_stmt->as<IntegerOffsetStmt>()->input = stmt;
      modifier.insert_after(stmt, std::move(offset_stmt));
      return;
    }

    set_done(stmt);
  }

  void visit(WhileControlStmt *stmt) override {
    if (stmt->mask) {
      stmt->mask = nullptr;
      modifier.mark_as_modified();
      return;
    }
  }

  static bool is_global_write(Stmt *stmt) {
    return stmt->is<GlobalStoreStmt>() || stmt->is<AtomicOpStmt>();
  }

  static bool is_atomic_value_used(const stmt_vector &clause,
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
    auto flatten = [&](stmt_vector &clause, bool true_branch) {
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
            auto load = Stmt::make<LocalLoadStmt>(store->dest);
            modifier.type_check(load.get(), config);
            auto select = Stmt::make<TernaryOpStmt>(
                TernaryOpType::select, if_stmt->cond,
                true_branch ? store->val : load.get(),
                true_branch ? load.get() : store->val);
            modifier.type_check(select.get(), config);
            store->val = select.get();
            modifier.insert_before(if_stmt, std::move(load));
            modifier.insert_before(if_stmt, std::move(select));
            modifier.insert_before(if_stmt, std::move(clause[i]));
          } else {
            modifier.insert_before(if_stmt, std::move(clause[i]));
          }
        }
        auto clean_clause = stmt_vector();
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
        modifier.mark_as_modified();
        return;
      }
      if (if_stmt->false_statements &&
          flatten(if_stmt->false_statements->statements, false)) {
        modifier.mark_as_modified();
        return;
      }
    }

    if (if_stmt->true_statements) {
      if (if_stmt->true_statements->statements.empty()) {
        if_stmt->set_true_statements(nullptr);
        modifier.mark_as_modified();
        return;
      }
    }

    if (if_stmt->false_statements) {
      if (if_stmt->false_statements->statements.empty()) {
        if_stmt->set_false_statements(nullptr);
        modifier.mark_as_modified();
        return;
      }
    }

    if (!if_stmt->true_statements && !if_stmt->false_statements) {
      modifier.erase(if_stmt);
      return;
    }

    if (config.advanced_optimization) {
      // Merge adjacent if's with the identical condition.
      // TODO: What about IfStmt::true_mask and IfStmt::false_mask?
      if (current_stmt_id < block->size() - 1 &&
          block->statements[current_stmt_id + 1]->is<IfStmt>()) {
        auto bstmt = block->statements[current_stmt_id + 1]->as<IfStmt>();
        if (bstmt->cond == if_stmt->cond) {
          auto concatenate = [](std::unique_ptr<Block> &clause1,
                                std::unique_ptr<Block> &clause2) {
            if (clause1 == nullptr) {
              clause1 = std::move(clause2);
              return;
            }
            if (clause2 != nullptr)
              clause1->insert(VecStatement(std::move(clause2->statements)), 0);
          };
          concatenate(bstmt->true_statements, if_stmt->true_statements);
          concatenate(bstmt->false_statements, if_stmt->false_statements);
          modifier.erase(if_stmt);
          return;
        }
      }
    }
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->has_body() && stmt->body->statements.empty()) {
      modifier.erase(stmt);
      return;
    }
  }

  void visit(WhileStmt *stmt) override {
    if (stmt->mask) {
      stmt->mask = nullptr;
      modifier.mark_as_modified();
      return;
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
    if (BasicBlockSimplify::run(block, visited, current_struct_for, config)) {
      modified = true;
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
      if (config.opt_level > 0 && whole_kernel_cse(root))
        modified = true;
      // Don't do this time-consuming optimization pass again if the IR is
      // not modified.
      if (config.opt_level > 0 && (first_iteration || modified) &&
          config.cfg_optimization &&
          cfg_optimization(root, args.after_lower_access,
                           args.autodiff_enabled))
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
