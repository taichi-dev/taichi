#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/transforms/lower_access.h"
#include "taichi/transforms/scalar_pointer_lowerer.h"

#include <deque>
#include <set>

namespace taichi {
namespace lang {
namespace {

class LowerAccess;

class PtrLowererImpl : public ScalarPointerLowerer {
 public:
  using ScalarPointerLowerer::ScalarPointerLowerer;

  void set_lower_access(LowerAccess *la);

  void set_pointer_needs_activation(bool v) {
    pointer_needs_activation_ = v;
  }

 protected:
  Stmt *handle_snode_at_level(int level,
                              LinearizeStmt *linearized,
                              Stmt *last) override;

 private:
  LowerAccess *la_{nullptr};
  std::unordered_set<SNode *> snodes_on_loop_;
  bool pointer_needs_activation_{false};
};

// Lower GlobalPtrStmt into smaller pieces for access optimization

class LowerAccess : public IRVisitor {
 public:
  DelayedIRModifier modifier;
  StructForStmt *current_struct_for;
  const std::vector<SNode *> &kernel_forces_no_activate;
  bool lower_atomic_ptr;
  bool packed;

  LowerAccess(const std::vector<SNode *> &kernel_forces_no_activate,
              bool lower_atomic_ptr,
              bool packed)
      : kernel_forces_no_activate(kernel_forces_no_activate),
        lower_atomic_ptr(lower_atomic_ptr),
        packed(packed) {
    // TODO: change this to false
    allow_undefined_visitor = true;
    current_struct_for = nullptr;
  }

  void visit(Block *stmt_list) override {
    for (auto &stmt : stmt_list->statements) {
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

  void visit(OffloadedStmt *stmt) override {
    stmt->all_blocks_accept(this);
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    current_struct_for = for_stmt;
    for_stmt->body->accept(this);
    current_struct_for = nullptr;
  }

  VecStatement lower_ptr(GlobalPtrStmt *ptr,
                         bool activate,
                         SNodeOpType snode_op = SNodeOpType::undefined) {
    VecStatement lowered;
    if (snode_op == SNodeOpType::is_active) {
      // For ti.is_active
      TI_ASSERT(!activate);
    }
    PtrLowererImpl lowerer{ptr->snode, ptr->indices,
                           snode_op,   ptr->is_bit_vectorized,
                           &lowered,   packed};
    lowerer.set_pointer_needs_activation(activate);
    lowerer.set_lower_access(this);
    lowerer.run();
    TI_ASSERT(lowered.size() > 0);
    auto lowered_ptr = lowered.back().get();
    if (ptr->is_bit_vectorized) {
      // if the global ptr is bit vectorized, we start from the place snode
      // and find the parent quant array snode, use its physical type
      auto parent_ret_type = ptr->snode->parent->physical_type;
      auto ptr_ret_type =
          TypeFactory::get_instance().get_pointer_type(parent_ret_type);
      lowered_ptr->ret_type = DataType(ptr_ret_type);
    } else {
      lowered_ptr->ret_type = ptr->snode->dt;
    }
    return lowered;
  }

  void visit(GlobalLoadStmt *stmt) override {
    if (!stmt->src->is<GlobalPtrStmt>())
      return;
    // No need to activate for all read accesses
    auto lowered = lower_ptr(stmt->src->as<GlobalPtrStmt>(), false);
    stmt->src = lowered.back().get();
    modifier.insert_before(stmt, std::move(lowered));
  }

  // TODO: this seems to be redundant
  void visit(PtrOffsetStmt *stmt) override {
    if (!stmt->is_unlowered_global_ptr())
      return;
    auto ptr = stmt->origin->as<GlobalPtrStmt>();
    // If ptr already has activate = false, no need to activate all the
    // generated micro-access ops. Otherwise, activate the nodes.
    auto lowered = lower_ptr(ptr, ptr->activate);
    stmt->origin = lowered.back().get();
    modifier.insert_before(stmt, std::move(lowered));
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (!stmt->dest->is<GlobalPtrStmt>())
      return;
    auto ptr = stmt->dest->as<GlobalPtrStmt>();
    // If ptr already has activate = false, no need to activate all the
    // generated micro-access ops. Otherwise, activate the nodes.
    auto lowered = lower_ptr(ptr, ptr->activate);
    stmt->dest = lowered.back().get();
    modifier.insert_before(stmt, std::move(lowered));
  }

  void visit(SNodeOpStmt *stmt) override {
    if (stmt->ptr->is<GlobalPtrStmt>()) {
      if (SNodeOpStmt::activation_related(stmt->op_type) &&
          stmt->snode->type != SNodeType::dynamic) {
        auto lowered =
            lower_ptr(stmt->ptr->as<GlobalPtrStmt>(), false, stmt->op_type);
        modifier.replace_with(stmt, std::move(lowered), true);
      } else if (stmt->op_type == SNodeOpType::get_addr) {
        auto lowered = lower_ptr(stmt->ptr->as<GlobalPtrStmt>(), false);
        auto cast = lowered.push_back<UnaryOpStmt>(UnaryOpType::cast_bits,
                                                   lowered.back().get());
        cast->cast_type = TypeFactory::get_instance().get_primitive_type(
            PrimitiveTypeID::u64);
        stmt->ptr = lowered.back().get();
        modifier.replace_with(stmt, std::move(lowered));
      } else {
        auto lowered = lower_ptr(stmt->ptr->as<GlobalPtrStmt>(),
                                 SNodeOpStmt::need_activation(stmt->op_type));
        stmt->ptr = lowered.back().get();
        modifier.insert_before(stmt, std::move(lowered));
      }
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    if (!lower_atomic_ptr)
      return;
    if (stmt->dest->is<GlobalPtrStmt>()) {
      auto lowered = lower_ptr(stmt->dest->as<GlobalPtrStmt>(),
                               stmt->dest->as<GlobalPtrStmt>()->activate);
      stmt->dest = lowered.back().get();
      modifier.insert_before(stmt, std::move(lowered));
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    if (stmt->val->is<GlobalPtrStmt>()) {
      auto lowered = lower_ptr(stmt->val->as<GlobalPtrStmt>(), true);
      stmt->val = lowered.back().get();
      modifier.insert_before(stmt, std::move(lowered));
    }
  }

  static bool run(IRNode *node,
                  const std::vector<SNode *> &kernel_forces_no_activate,
                  bool lower_atomic,
                  bool packed) {
    LowerAccess inst(kernel_forces_no_activate, lower_atomic, packed);
    bool modified = false;
    while (true) {
      node->accept(&inst);
      if (inst.modifier.modify_ir()) {
        modified = true;
      } else {
        break;
      }
    }
    return modified;
  }
};

void PtrLowererImpl::set_lower_access(LowerAccess *la) {
  la_ = la;

  snodes_on_loop_.clear();
  if (la_->current_struct_for) {
    for (SNode *s = la_->current_struct_for->snode; s != nullptr;
         s = s->parent) {
      snodes_on_loop_.insert(s);
    }
  }
}

Stmt *PtrLowererImpl::handle_snode_at_level(int level,
                                            LinearizeStmt *linearized,
                                            Stmt *last) {
  // Check whether |snode| is part of the tree being iterated over by struct for
  auto *snode = snodes()[level];
  bool on_loop_tree = (snodes_on_loop_.find(snode) != snodes_on_loop_.end());
  auto *current_struct_for = la_->current_struct_for;
  if (on_loop_tree && current_struct_for &&
      (indices_.size() == current_struct_for->snode->num_active_indices)) {
    for (int j = 0; j < (int)indices_.size(); j++) {
      auto diff = irpass::analysis::value_diff_loop_index(
          indices_[j], current_struct_for, j);
      if (!diff.linear_related()) {
        on_loop_tree = false;
      } else if (j == (int)indices_.size() - 1) {
        if (!(0 <= diff.low && diff.high <= 1)) {  // TODO: Vectorize
          on_loop_tree = false;
        }
      } else {
        if (!diff.certain() || diff.low != 0) {
          on_loop_tree = false;
        }
      }
    }
  }

  // Generates the SNode access operations at the current |level|.
  if ((snode_op_ != SNodeOpType::undefined) &&
      (level == (int)snodes().size() - 1)) {
    // Create a SNodeOp querying if element i(linearized) of node is active
    lowered_->push_back<SNodeOpStmt>(snode_op_, snode, last, linearized);
  } else {
    const bool kernel_forces_no_activate_snode =
        std::find(la_->kernel_forces_no_activate.begin(),
                  la_->kernel_forces_no_activate.end(),
                  snode) != la_->kernel_forces_no_activate.end();

    const bool needs_activation =
        snode->need_activation() && pointer_needs_activation_ &&
        !kernel_forces_no_activate_snode && !on_loop_tree;

    auto lookup = lowered_->push_back<SNodeLookupStmt>(snode, last, linearized,
                                                       needs_activation);
    int chid = snode->child_id(snodes()[level + 1]);
    if (is_bit_vectorized_ && (snode->type == SNodeType::dense) &&
        (level == path_length() - 2)) {
      last = lowered_->push_back<GetChStmt>(lookup, chid,
                                            /*is_bit_vectorized=*/true);
    } else {
      last = lowered_->push_back<GetChStmt>(lookup, chid,
                                            /*is_bit_vectorized=*/false);
    }
  }
  return last;
}

}  // namespace

const PassID LowerAccessPass::id = "LowerAccessPass";

namespace irpass {

bool lower_access(IRNode *root,
                  const CompileConfig &config,
                  const LowerAccessPass::Args &args) {
  bool modified = LowerAccess::run(root, args.kernel_forces_no_activate,
                                   args.lower_atomic, config.packed);
  type_check(root, config);
  return modified;
}

}  // namespace irpass
}  // namespace lang
}  // namespace taichi
