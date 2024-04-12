#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/utils.h"

#include <typeinfo>
#include <algorithm>

namespace taichi::lang {

template <typename T>
Stmt *insert_const(const DataType &dtype,
                   Stmt *stmt,
                   const T &value,
                   bool insert_before_me = false) {
  auto type = dtype.ptr_removed();
  Stmt *zero = nullptr;
  if (insert_before_me)
    zero = stmt->insert_before_me(
        Stmt::make<ConstStmt>(TypedConstant(type.get_element_type(), value)));
  else
    zero = stmt->insert_after_me(
        Stmt::make<ConstStmt>(TypedConstant(type.get_element_type(), value)));

  if (type->is<TensorType>()) {
    auto t_dtype = type->as<TensorType>();
    std::vector<Stmt *> values(t_dtype->get_num_elements(), zero);
    if (insert_before_me) {
      zero = zero->insert_before_me(Stmt::make<MatrixInitStmt>(values));
    } else {
      zero = zero->insert_after_me(Stmt::make<MatrixInitStmt>(values));
    }
    zero->ret_type = type;
  }
  return zero;
}

class IndependentBlockMetaData {
 public:
  bool is_ib = true;
  bool is_smallest_ib = true;
};

class NonLinearOps {
 public:
  inline static const std::set<TernaryOpType> ternary_collections{
      TernaryOpType::select};
  inline static const std::set<UnaryOpType> unary_collections{
      UnaryOpType::abs,  UnaryOpType::sin,  UnaryOpType::cos,
      UnaryOpType::tanh, UnaryOpType::asin, UnaryOpType::acos,
      UnaryOpType::exp,  UnaryOpType::log,  UnaryOpType::sqrt};
  inline static const std::set<BinaryOpType> binary_collections{
      BinaryOpType::mul, BinaryOpType::div, BinaryOpType::atan2,
      BinaryOpType::pow};
};

class IndependentBlocksJudger : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void visit(LocalLoadStmt *stmt) override {
    TI_ASSERT(stmt->src->is<AllocaStmt>() || stmt->src->is<MatrixPtrStmt>() ||
              stmt->src->is<MatrixOfMatrixPtrStmt>());
    touched_allocas_.insert(stmt->src);
  }

  void visit(LocalStoreStmt *stmt) override {
    TI_ASSERT(stmt->dest->is<AllocaStmt>() || stmt->dest->is<MatrixPtrStmt>() ||
              stmt->dest->is<MatrixOfMatrixPtrStmt>());
    touched_allocas_.insert(stmt->dest);
  }

  void visit(AtomicOpStmt *stmt) override {
    // We don't need to check the global atomics inside the range for-loops
    // because
    // 1. If the range for-loop is innermost, they will be captured by
    // MakeAdjoint anyway
    // 2. If the range for-loop is not innermost, they will be processed by
    // another IndependentBlocksJudger
    if (is_inside_loop_)
      return;

    Stmt *dest = stmt->dest;
    if (dest->is<MatrixPtrStmt>()) {
      dest = dest->as<MatrixPtrStmt>()->origin;
    }

    if (dest->is<ExternalPtrStmt>()) {
      if (dest->as<ExternalPtrStmt>()
              ->base_ptr->as<ArgLoadStmt>()
              ->ret_type.ptr_removed()
              ->as<StructType>()
              ->elements()
              .size() > TypeFactory::GRAD_PTR_POS_IN_NDARRAY) {
        qualified_glb_operations_ = true;
      }
    } else {
      TI_ASSERT(dest->is<GlobalPtrStmt>());
      if (dest->as<GlobalPtrStmt>()->snode->has_adjoint()) {
        qualified_glb_operations_ = true;
      }
    }
  }

  void visit(GlobalLoadStmt *stmt) override {
    // We don't need to check the global load inside the range for-loops
    // because
    // 1. If the range for-loop is innermost, they will be captured by
    // MakeAdjoint anyway
    // 2. If the range for-loop is not innermost, they will be processed by
    // another IndependentBlocksJudger
    if (is_inside_loop_)
      return;

    Stmt *src = stmt->src;
    if (src->is<MatrixPtrStmt>()) {
      src = src->as<MatrixPtrStmt>()->origin;
    }

    if ((src->is<ExternalPtrStmt>() &&
         src->as<ExternalPtrStmt>()
                 ->base_ptr->as<ArgLoadStmt>()
                 ->ret_type.ptr_removed()
                 ->as<StructType>()
                 ->elements()
                 .size() > TypeFactory::GRAD_PTR_POS_IN_NDARRAY) ||
        (src->is<GlobalPtrStmt>() &&
         src->as<GlobalPtrStmt>()->snode->has_adjoint())) {
      qualified_glb_operations_ = true;
    }
  }

  void visit(RangeForStmt *stmt) override {
    inner_most_loop_ = false;
    is_inside_loop_ = true;
    stmt->body->accept(this);
    is_inside_loop_ = false;
  }

  static void run(IRNode *root, IndependentBlockMetaData &ib_meta_data) {
    IndependentBlocksJudger Judger;
    Block *block = root->as<Block>();
    root->accept(&Judger);
    std::set<Block *> outside_blocks;
    // Collect all parent blocks (i.e. outside blocks) of the current block for
    // local load/store stmt checks
    for (auto b = block->parent_block(); b; b = b->parent_block()) {
      if (b)
        outside_blocks.insert(b);
    }
    for (const auto &alloca : Judger.touched_allocas_) {
      // Test if the alloca belongs to the current block
      if (outside_blocks.find(alloca->parent) != outside_blocks.end()) {
        // This block is not an IB since it loads/modifies outside variables
        ib_meta_data.is_ib = false;
      }
    }

    // To judge whether a block is an IB
    // - No local load/store to allocas *outside* itself has been strictly
    // enforced

    // To judge whether a block is a smallest IB
    // - If the #1 is satisfied, either an inner most loop or a block without
    // global atomics / global load is an IB
    ib_meta_data.is_smallest_ib =
        ib_meta_data.is_ib &&
        (Judger.qualified_glb_operations_ || Judger.inner_most_loop_);
  }

 private:
  std::set<Stmt *> touched_allocas_;
  bool qualified_glb_operations_ = false;
  bool inner_most_loop_ = true;
  bool is_inside_loop_ = false;
};

// Remove the duplicated IBs, remove blocks who are others' children because
// each block should only be processed once
class DuplicateIndependentBlocksCleaner : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void check_children_ib(Block *target_block) {
    // Remove the block if it is the child of the block being visiting
    if (independent_blocks_cleaned_.find(target_block) !=
        independent_blocks_cleaned_.end()) {
      independent_blocks_cleaned_.erase(target_block);
    }
  }

  void visit(StructForStmt *stmt) override {
    check_children_ib(stmt->body.get());
    stmt->body->accept(this);
  }
  void visit(RangeForStmt *stmt) override {
    check_children_ib(stmt->body.get());
    stmt->body->accept(this);
  }

  static std::set<Block *> run(
      const std::vector<std::pair<int, Block *>> &raw_IBs) {
    DuplicateIndependentBlocksCleaner cleaner;
    // Remove duplicate IBs
    for (auto const &item : raw_IBs) {
      cleaner.independent_blocks_cleaned_.insert(item.second);
    }
    // No clean is needed if only one IB exists
    if (cleaner.independent_blocks_cleaned_.size() > 1) {
      // Check from the block with smallest depth, ensure no duplicate visit
      // happens
      for (const auto &block : cleaner.independent_blocks_cleaned_) {
        block->accept(&cleaner);
      }
    }
    return cleaner.independent_blocks_cleaned_;
  }

 private:
  std::set<Block *> independent_blocks_cleaned_;
};

// Do automatic differentiation pass in the reverse order (reverse-mode AD)

// Independent Block (IB): blocks (i.e. loop bodies) whose iterations are
// independent of previous iterations and outer scopes. IBs are where the
// MakeAdjoint pass happens. IBs may contain if's and for-loops.

// IBs are not always the inner-most for loop body. If the inner-most for-loop
// has iterations that carry iteration-dependent variables, it's not an IB.

// Clearly the outermost level is always an IB, but we want IBs to be as small
// as possible. Outside IBs, we just need to reverse the for-loop orders.

// Figure out the IBs.
class IdentifyIndependentBlocks : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void visit(WhileStmt *stmt) override {
    TI_ERROR("WhileStmt is not supported in AutoDiff.");
  }

  void visit(ContinueStmt *stmt) override {
    TI_ERROR("ContinueStmt is not supported in AutoDiff.");
  }

  void visit(WhileControlStmt *stmt) override {
    TI_ERROR("WhileControlStmt (break) is not supported in AutoDiff.");
  }

  void visit_loop_body(Block *block) {
    auto ib_meta_data = IndependentBlockMetaData();
    // An IB has no local load/store to allocas *outside* itself
    // Note:
    //  - Local atomics should have been demoted before this pass.
    //  - It is OK for an IB to have more than two for loops.
    //  - No global load/atomics operations to the global variables which
    //  require gradient
    if (block->statements.empty()) {
      // A empty block shoud be a smallest IB
      ib_meta_data.is_ib = true;
      ib_meta_data.is_smallest_ib = true;
    } else {
      IndependentBlocksJudger::run(block, ib_meta_data);
    }

    if (ib_meta_data.is_smallest_ib) {
      independent_blocks_.push_back({depth_, block});
    } else if (ib_meta_data.is_ib) {
      current_ib_ = block;
      block->accept(this);
    } else {
      if (depth_ <= 1) {
        TI_ASSERT(depth_ == 1);
        // The top level block is already not an IB, store it
        independent_blocks_.push_back({depth_ - 1, block});
      } else {
        independent_blocks_.push_back({depth_ - 1, block->parent_block()});
      }
    }
  }

  void visit(StructForStmt *stmt) override {
    TI_ASSERT(depth_ == 0);
    depth_++;
    current_ib_ = stmt->body.get();
    visit_loop_body(stmt->body.get());
    depth_--;
  }

  void visit(RangeForStmt *stmt) override {
    if (depth_ == 0) {
      current_ib_ = stmt->body.get();
    }
    depth_++;
    visit_loop_body(stmt->body.get());
    depth_--;
  }

  static std::set<Block *> run(IRNode *root) {
    IdentifyIndependentBlocks pass;
    Block *block = root->as<Block>();
    bool has_for = false;
    for (auto &s : block->statements) {
      if (s->is<StructForStmt>() || s->is<RangeForStmt>()) {
        has_for = true;
      }
    }
    if (!has_for) {
      // The whole block is an IB
      pass.independent_blocks_.push_back({0, block});
    } else {
      root->accept(&pass);
    }
    // Sort the IBs by their depth from shallow to deep
    std::sort(pass.independent_blocks_.begin(), pass.independent_blocks_.end(),
              [](const std::pair<int, Block *> &a,
                 const std::pair<int, Block *> &b) -> bool {
                return a.first < b.first;
              });

    TI_ASSERT(!pass.independent_blocks_.empty());
    return DuplicateIndependentBlocksCleaner::run(pass.independent_blocks_);
  }

 private:
  std::vector<std::pair<int, Block *>> independent_blocks_;
  int depth_{0};
  Block *current_ib_{nullptr};
};

// Note that SSA does not mean the instruction will be executed at most once.
// For instructions that may be executed multiple times, we treat them as a
// mutable local variables.
class PromoteSSA2LocalVar : public BasicStmtVisitor {
  using BasicStmtVisitor::visit;

  explicit PromoteSSA2LocalVar(Block *block) {
    alloca_block_ = block;
    invoke_default_visitor = true;
    execute_once_ = true;
  }

  void visit(Stmt *stmt) override {
    if (execute_once_)
      return;
    if (!(stmt->is<UnaryOpStmt>() || stmt->is<BinaryOpStmt>() ||
          stmt->is<TernaryOpStmt>() || stmt->is<GlobalLoadStmt>() ||
          stmt->is<LoopIndexStmt>() || stmt->is<AllocaStmt>())) {
      // TODO: this list may be incomplete
      return;
    }

    if (stmt->is<AllocaStmt>()) {
      // Create a new alloc at the top of an ib to replace the old alloca
      auto dtype = stmt->ret_type.ptr_removed();
      auto alloc = Stmt::make<AllocaStmt>(dtype);
      auto alloc_ptr = alloc.get();
      TI_ASSERT(alloca_block_);
      alloca_block_->insert(std::move(alloc), 0);
      // Replace all the usages of the old stmt with that of the new one
      irpass::replace_all_usages_with(stmt->parent, stmt, alloc_ptr);

      // Replace the old alloca with a local store
      // and it will be replaced by a AdStackPushStmt in the following
      // ReplaceLocalVarWithStacks pass

      auto zero = insert_const(dtype, stmt, 0);
      zero->insert_after_me(Stmt::make<LocalStoreStmt>(alloc_ptr, zero));
      // Remove the old stmt
      stmt->parent->erase(stmt);
    } else {
      // Create a alloc
      auto alloc = Stmt::make<AllocaStmt>(stmt->ret_type.ptr_removed());
      auto alloc_ptr = alloc.get();
      TI_ASSERT(alloca_block_);
      alloca_block_->insert(std::move(alloc), 0);
      auto load = stmt->insert_after_me(Stmt::make<LocalLoadStmt>(alloc_ptr));
      irpass::replace_all_usages_with(stmt->parent, stmt, load);
      // Create the load first so that the operand of the store won't get
      // replaced
      stmt->insert_after_me(Stmt::make<LocalStoreStmt>(alloc_ptr, stmt));
    }
  }

  void visit(RangeForStmt *stmt) override {
    auto old_execute_once = execute_once_;
    execute_once_ = false;  // loop body may be executed many times
    stmt->body->accept(this);
    execute_once_ = old_execute_once;
  }

 private:
  Block *alloca_block_{nullptr};
  bool execute_once_;

 public:
  static void run(Block *block) {
    PromoteSSA2LocalVar pass(block);
    block->accept(&pass);
  }
};

class AdStackAllocaJudger : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  // Find the usage of the stmt recursively along the LocalLoadStmt
  void visit(LocalLoadStmt *stmt) override {
    if (stmt->src == target_alloca_) {
      local_loaded_ = true;
      target_alloca_ = stmt;
    }
  }

  // Check if there is a LocalLoadStmt - LocalStoreStmt cycle for an alloca
  // Check if the alloca is load only
  void visit(LocalStoreStmt *stmt) override {
    if (stmt->dest == target_alloca_backup_)
      load_only_ = false;
    if (local_loaded_ && stmt->dest == target_alloca_backup_) {
      is_stack_needed_ = true;
    }
  }

  // Check if the alloca is load only
  void visit(AtomicOpStmt *stmt) override {
    if (stmt->dest == target_alloca_backup_)
      load_only_ = false;
  }

  // The stack is needed if the alloc serves as the index of any global
  // variables
  void visit(GlobalPtrStmt *stmt) override {
    if (is_stack_needed_)
      return;
    for (const auto &index : stmt->indices) {
      if (index == target_alloca_)
        is_stack_needed_ = true;
    }
  }

  void visit(ExternalPtrStmt *stmt) override {
    if (is_stack_needed_)
      return;
    for (const auto &index : stmt->indices) {
      if (index == target_alloca_)
        is_stack_needed_ = true;
    }
  }

  // Check whether the target stmt is used by the UnaryOpStmts who requires the
  // ad stack
  void visit(UnaryOpStmt *stmt) override {
    if (is_stack_needed_)
      return;
    if (NonLinearOps::unary_collections.find(stmt->op_type) !=
        NonLinearOps::unary_collections.end()) {
      if (stmt->operand == target_alloca_)
        is_stack_needed_ = true;
    }
  }

  // Check whether the target stmt is used by the BinaryOpStmts who requires the
  // ad stack
  void visit(BinaryOpStmt *stmt) override {
    if (is_stack_needed_)
      return;
    if (NonLinearOps::binary_collections.find(stmt->op_type) !=
        NonLinearOps::binary_collections.end()) {
      if (stmt->lhs == target_alloca_ || stmt->rhs == target_alloca_)
        is_stack_needed_ = true;
    }
  }

  // Check whether the target stmt is used by the TernaryOpStmts who requires
  // the ad stack
  void visit(TernaryOpStmt *stmt) override {
    if (is_stack_needed_)
      return;
    if (NonLinearOps::ternary_collections.find(stmt->op_type) !=
        NonLinearOps::ternary_collections.end()) {
      if (stmt->op1 == target_alloca_ || stmt->op2 == target_alloca_ ||
          stmt->op3 == target_alloca_)
        is_stack_needed_ = true;
    }
  }

  // Check whether the target serves as the condition of a if stmt
  void visit(IfStmt *stmt) override {
    if (is_stack_needed_)
      return;

    if (stmt->cond == target_alloca_) {
      is_stack_needed_ = true;
      return;
    }

    if (stmt->true_statements)
      stmt->true_statements->accept(this);
    if (stmt->false_statements)
      stmt->false_statements->accept(this);
  }

  static bool run(AllocaStmt *target_alloca) {
    AdStackAllocaJudger judger;
    judger.target_alloca_ = target_alloca;
    judger.target_alloca_backup_ = target_alloca;
    target_alloca->parent->accept(&judger);
    return (!judger.load_only_) && judger.is_stack_needed_;
  }

 private:
  Stmt *target_alloca_;
  Stmt *target_alloca_backup_;
  bool is_stack_needed_ = false;
  bool local_loaded_ = false;
  bool load_only_ = true;
};

class RegulateTensorTypedStatements : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier delayed_modifier_;

  explicit RegulateTensorTypedStatements() {
  }

  template <typename Store, typename Load>
  void process_store_stmt(Store *stmt) {
    TI_ASSERT(stmt->template is<LocalStoreStmt>() ||
              stmt->template is<GlobalStoreStmt>());

    if (stmt->dest->template is<MatrixPtrStmt>()) {
      auto matrix_ptr_stmt = stmt->dest->template as<MatrixPtrStmt>();
      auto orig_stmt = matrix_ptr_stmt->origin;

      if (!orig_stmt->ret_type.ptr_removed()->template is<TensorType>()) {
        return;
      }

      auto tensor_type =
          orig_stmt->ret_type.ptr_removed()->template as<TensorType>();
      auto num_elements = tensor_type->get_num_elements();

      if (matrix_ptr_stmt->offset->template is<ConstStmt>()) {
        /*
          [Static index]
          Fwd:
          $0 = alloca <4 x i32>
          $1 = load $0
          $2 = matrix ptr $1, 2 // offset = 2
          $3 : local store $2, $val

          Replaced:
          $0 = alloca <4 x i32>
          $1 = load $0
          $2 = matrix ptr $1, 2 // --> erase

          $3 = matrix ptr $1, 0
          $4 = load $3

          $5 = matrix ptr $1, 1
          $6 = load $5

          $7 = matrix ptr $1, 3
          $8 = load $7

          $9 = matrix init [$4, $6, $val, $8]

          $10 : store $0, $9
        */
        int offset =
            matrix_ptr_stmt->offset->template as<ConstStmt>()->val.val_int32();

        TI_ASSERT(offset < num_elements);

        std::vector<Stmt *> values;
        for (int i = 0; i < num_elements; i++) {
          if (i == offset) {
            values.push_back(stmt->val);
            continue;
          }

          auto const_i = insert_const(PrimitiveType::i32, stmt, i, true);
          auto matrix_ptr_stmt_i =
              Stmt::make<MatrixPtrStmt>(orig_stmt, const_i);
          matrix_ptr_stmt_i->ret_type = tensor_type->get_element_type();

          auto local_load_stmt_i = Stmt::make<Load>(matrix_ptr_stmt_i.get());
          local_load_stmt_i->ret_type = tensor_type->get_element_type();

          values.push_back(local_load_stmt_i.get());

          stmt->insert_before_me(std::move(matrix_ptr_stmt_i));
          stmt->insert_before_me(std::move(local_load_stmt_i));
        }

        auto matrix_init_stmt = Stmt::make<MatrixInitStmt>(values);
        matrix_init_stmt->ret_type = tensor_type;

        auto store_stmt = Stmt::make<Store>(orig_stmt, matrix_init_stmt.get());
        stmt->insert_before_me(std::move(matrix_init_stmt));
        stmt->replace_with(std::move(store_stmt));

        return;

      } else {
        /*
          [Dynamic index]
          Fwd:
          $0 = alloca <4 x i32>
          $1 = load $0
          $2 = matrix ptr $1, $offset // offset = 2
          $3 : local store $2, $val

          Replaced:
          $0 = alloca <4 x i32>

          $1 = load $0
          $2 = matrix init [$val, $val, $val, $val]

          $3 = matrix init [$offset, $offset, $offset, $offset]
          $4 = matrix init [0, 1, 2, 3]

          $5 = bin_eq $3, $4
          $6 = select $5, $2, $1

          $7 : store $0, $6
        */
        auto tensor_type =
            orig_stmt->ret_type.ptr_removed()->template as<TensorType>();
        auto num_elements = tensor_type->get_num_elements();

        auto tensor_shape = tensor_type->get_shape();
        auto index_tensor_type = TypeFactory::get_instance().get_tensor_type(
            tensor_shape, PrimitiveType::i32);

        std::vector<Stmt *> val_values(num_elements, stmt->val);
        std::vector<Stmt *> offset_values(num_elements,
                                          matrix_ptr_stmt->offset);
        std::vector<Stmt *> index_values(num_elements);
        for (int i = 0; i < num_elements; i++) {
          index_values[i] = insert_const(PrimitiveType::i32, stmt, i, true);
        }

        auto matrix_val = Stmt::make<MatrixInitStmt>(val_values);
        matrix_val->ret_type = tensor_type;

        auto matrix_offset = Stmt::make<MatrixInitStmt>(offset_values);
        matrix_offset->ret_type = index_tensor_type;

        auto matrix_index = Stmt::make<MatrixInitStmt>(index_values);
        matrix_index->ret_type = index_tensor_type;
        auto cmp_tensor_type = TypeFactory::get_instance().get_tensor_type(
            tensor_shape, PrimitiveType::u1);
        auto matrix_eq = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_eq, matrix_offset.get(), matrix_index.get());
        matrix_eq->ret_type = cmp_tensor_type;

        auto orig_value = Stmt::make<Load>(orig_stmt);
        orig_value->ret_type = tensor_type;

        auto matrix_select =
            Stmt::make<TernaryOpStmt>(TernaryOpType::select, matrix_eq.get(),
                                      matrix_val.get(), orig_value.get());
        matrix_select->ret_type = tensor_type;

        auto store_stmt = Stmt::make<Store>(orig_stmt, matrix_select.get());

        stmt->insert_before_me(std::move(matrix_val));
        stmt->insert_before_me(std::move(matrix_offset));
        stmt->insert_before_me(std::move(matrix_index));
        stmt->insert_before_me(std::move(matrix_eq));
        stmt->insert_before_me(std::move(orig_value));
        stmt->insert_before_me(std::move(matrix_select));
        stmt->replace_with(std::move(store_stmt));
        return;
      }
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    process_store_stmt<LocalStoreStmt, LocalLoadStmt>(stmt);
  }

  void visit(GlobalStoreStmt *stmt) override {
    process_store_stmt<GlobalStoreStmt, GlobalLoadStmt>(stmt);
  }

  static void run(IRNode *root) {
    RegulateTensorTypedStatements pass;
    root->accept(&pass);
  }
};

class ReplaceLocalVarWithStacks : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  int ad_stack_size;
  DelayedIRModifier delayed_modifier_;

  explicit ReplaceLocalVarWithStacks(int ad_stack_size)
      : ad_stack_size(ad_stack_size) {
  }

  void visit(AllocaStmt *alloc) override {
    bool is_stack_needed = AdStackAllocaJudger::run(alloc);
    if (is_stack_needed) {
      auto dtype = alloc->ret_type.ptr_removed();
      auto stack_alloca = Stmt::make<AdStackAllocaStmt>(dtype, ad_stack_size);
      auto stack_alloca_ptr = stack_alloca.get();

      alloc->replace_with(VecStatement(std::move(stack_alloca)));

      // Note that unlike AllocaStmt, AdStackAllocaStmt does NOT have an 0 as
      // initial value. Therefore here we push an initial 0 value.
      auto zero = insert_const(dtype, stack_alloca_ptr, 0);
      zero->insert_after_me(
          Stmt::make<AdStackPushStmt>(stack_alloca_ptr, zero));
    }
  }

  void visit(LocalLoadStmt *stmt) override {
    if (stmt->src->is<AdStackAllocaStmt>()) {
      auto stack_load = Stmt::make<AdStackLoadTopStmt>(stmt->src);
      stack_load->ret_type = stmt->ret_type;

      stmt->replace_with(std::move(stack_load));
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    if (stmt->dest->is<MatrixPtrStmt>()) {
      auto matrix_ptr_stmt = stmt->dest->as<MatrixPtrStmt>();
      if (matrix_ptr_stmt->origin->is<AdStackLoadTopStmt>()) {
        auto stack_top_stmt = matrix_ptr_stmt->origin->as<AdStackLoadTopStmt>();
        TI_ASSERT(stack_top_stmt->return_ptr == true);

        if (!stack_top_stmt->ret_type.ptr_removed()->is<TensorType>()) {
          return;
        }

        auto tensor_type =
            stack_top_stmt->ret_type.ptr_removed()->as<TensorType>();
        auto num_elements = tensor_type->get_num_elements();

        if (matrix_ptr_stmt->offset->is<ConstStmt>()) {
          /*
            [Static index]
            Fwd:
            $1 = alloca <4 x i32>
            $2 = matrix ptr $1, 2 // offset = 2
            $3 : local store $2, $val

            Replaced:
            $1 =  alloca <4 x i32>
            $2 = matrix ptr $1, 2 // --> erase

            $3 = matrix ptr $1, 0
            $4 = load $3

            $5 = matrix ptr $1, 1
            $6 = load $5

            $7 = matrix ptr $1, 3
            $8 = load $7

            $9 = matrix init [$4, $6, $val, $8]

            $10 : store $1, $9
          */
          int offset =
              matrix_ptr_stmt->offset->as<ConstStmt>()->val.val_int32();

          TI_ASSERT(offset < num_elements);

          std::vector<Stmt *> values;
          for (int i = 0; i < num_elements; i++) {
            if (i == offset) {
              values.push_back(stmt->val);
              continue;
            }

            auto const_i = insert_const(PrimitiveType::i32, stmt, i, true);
            auto matrix_ptr_stmt_i =
                Stmt::make<MatrixPtrStmt>(stack_top_stmt, const_i);
            matrix_ptr_stmt_i->ret_type = tensor_type->get_element_type();

            auto local_load_stmt_i =
                Stmt::make<LocalLoadStmt>(matrix_ptr_stmt_i.get());
            local_load_stmt_i->ret_type = tensor_type->get_element_type();

            values.push_back(local_load_stmt_i.get());

            stmt->insert_before_me(std::move(matrix_ptr_stmt_i));
            stmt->insert_before_me(std::move(local_load_stmt_i));
          }

          auto matrix_init_stmt = Stmt::make<MatrixInitStmt>(values);
          matrix_init_stmt->ret_type = tensor_type;

          auto stack_push = Stmt::make<AdStackPushStmt>(stack_top_stmt->stack,
                                                        matrix_init_stmt.get());
          stmt->insert_before_me(std::move(matrix_init_stmt));
          stmt->replace_with(std::move(stack_push));

          return;

        } else {
          /*
            [Dynamic index]
            Fwd:
            $1 = alloca <4 x i32>
            $2 = matrix ptr $1, $offset // offset = 2
            $3 : local store $2, $val

            Replaced:
            $1 = alloca <4 x i32>

            $2 = matrix init [$val, $val, $val, $val]

            $3 = matrix init [$offset, $offset, $offset, $offset]
            $4 = matrix init [0, 1, 2, 3]

            $5 = bin_eq $3, $4
            $6 = select $5, $2, $1

            $7 : store $1, $6
          */
          auto tensor_type =
              stack_top_stmt->ret_type.ptr_removed()->as<TensorType>();
          auto num_elements = tensor_type->get_num_elements();

          auto tensor_shape = tensor_type->get_shape();
          auto index_tensor_type = TypeFactory::get_instance().get_tensor_type(
              tensor_shape, PrimitiveType::i32);

          std::vector<Stmt *> val_values(num_elements, stmt->val);
          std::vector<Stmt *> offset_values(num_elements,
                                            matrix_ptr_stmt->offset);
          std::vector<Stmt *> index_values(num_elements);
          for (int i = 0; i < num_elements; i++) {
            index_values[i] = insert_const(PrimitiveType::i32, stmt, i, true);
          }

          auto matrix_val = Stmt::make<MatrixInitStmt>(val_values);
          matrix_val->ret_type = tensor_type;

          auto matrix_offset = Stmt::make<MatrixInitStmt>(offset_values);
          matrix_offset->ret_type = index_tensor_type;

          auto matrix_index = Stmt::make<MatrixInitStmt>(index_values);
          matrix_index->ret_type = index_tensor_type;

          auto cmp_tensor_type = TypeFactory::get_instance().get_tensor_type(
              tensor_shape, PrimitiveType::u1);
          auto matrix_eq = Stmt::make<BinaryOpStmt>(
              BinaryOpType::cmp_eq, matrix_offset.get(), matrix_index.get());
          matrix_eq->ret_type = cmp_tensor_type;

          auto matrix_alloca_value =
              Stmt::make<AdStackLoadTopStmt>(stack_top_stmt->stack);
          matrix_alloca_value->ret_type = tensor_type;

          auto matrix_select = Stmt::make<TernaryOpStmt>(
              TernaryOpType::select, matrix_eq.get(), matrix_val.get(),
              matrix_alloca_value.get());
          matrix_select->ret_type = tensor_type;

          auto stack_push = Stmt::make<AdStackPushStmt>(stack_top_stmt->stack,
                                                        matrix_select.get());

          stmt->insert_before_me(std::move(matrix_val));
          stmt->insert_before_me(std::move(matrix_offset));
          stmt->insert_before_me(std::move(matrix_index));
          stmt->insert_before_me(std::move(matrix_eq));
          stmt->insert_before_me(std::move(matrix_alloca_value));
          stmt->insert_before_me(std::move(matrix_select));
          stmt->replace_with(std::move(stack_push));

          return;
        }
      }
    }

    // Non Tensor-type
    if (stmt->dest->is<AdStackAllocaStmt>())
      stmt->replace_with(Stmt::make<AdStackPushStmt>(stmt->dest, stmt->val));
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<AdStackAllocaStmt>()) {
      auto stack_top =
          Stmt::make<AdStackLoadTopStmt>(stmt->origin, true /*is_ptr*/);
      stack_top->ret_type = stmt->origin->ret_type;
      stack_top->ret_type.set_is_pointer(true);

      Stmt *stack_top_stmt = stack_top.get();
      stmt->insert_before_me(std::move(stack_top));

      auto new_matrix_ptr_stmt =
          Stmt::make<MatrixPtrStmt>(stack_top_stmt, stmt->offset);
      new_matrix_ptr_stmt->ret_type = stmt->ret_type;
      stmt->replace_with(std::move(new_matrix_ptr_stmt));
    }
  }
};

class ReverseOuterLoops : public BasicStmtVisitor {
  using BasicStmtVisitor::visit;

 private:
  explicit ReverseOuterLoops(const std::set<Block *> &IB)
      : loop_depth_(0), ib_(IB) {
  }

  bool is_ib(Block *block) const {
    return std::find(ib_.begin(), ib_.end(), block) != ib_.end();
  }

  void visit(StructForStmt *stmt) override {
    loop_depth_ += 1;
    if (!is_ib(stmt->body.get()))
      stmt->body->accept(this);
    loop_depth_ -= 1;
  }

  void visit(RangeForStmt *stmt) override {
    if (loop_depth_ >= 1) {
      stmt->reversed = !stmt->reversed;
    }
    loop_depth_ += 1;
    if (!is_ib(stmt->body.get()))
      stmt->body->accept(this);
    loop_depth_ -= 1;
  }

  int loop_depth_;
  std::set<Block *> ib_;

 public:
  static void run(IRNode *root, const std::set<Block *> &IB) {
    ReverseOuterLoops pass(IB);
    root->accept(&pass);
  }
};

// Base class for both reverse (make adjoint) and forward (make dual) mode
class ADTransform : public IRVisitor {
 protected:
  Stmt *constant(float32 x, DataType dtype = PrimitiveType::unknown) {
    dtype.set_is_pointer(false);
    if (!dtype->is<TensorType>())
      return insert<ConstStmt>(TypedConstant(x));

    auto tensor_type = dtype->as<TensorType>();
    auto num_elements = tensor_type->get_num_elements();
    std::vector<Stmt *> values;
    for (int i = 0; i < num_elements; i++) {
      values.push_back(insert<ConstStmt>(TypedConstant(x)));
    }
    auto matrix_init_stmt = insert<MatrixInitStmt>(values);
    matrix_init_stmt->ret_type = tensor_type;
    return matrix_init_stmt;
  }

  // utils
  Stmt *sgn(Stmt *inp) {
    return insert<UnaryOpStmt>(UnaryOpType::sgn, load(inp));
  }

  // utils
  Stmt *negate(Stmt *inp) {
    return insert<UnaryOpStmt>(UnaryOpType::neg, load(inp));
  }

  Stmt *sqrt(Stmt *inp) {
    return insert<UnaryOpStmt>(UnaryOpType::sqrt, load(inp));
  }

  Stmt *rsqrt(Stmt *inp) {
    return insert<UnaryOpStmt>(UnaryOpType::rsqrt, load(inp));
  }

  Stmt *mul(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::mul, load(op1), load(op2));
  }

  Stmt *sqr(Stmt *op1) {
    return mul(op1, op1);
  }

  Stmt *add(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::add, load(op1), load(op2));
  }

  Stmt *cmp_lt(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::cmp_lt, load(op1), load(op2));
  }

  Stmt *sub(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::sub, load(op1), load(op2));
  }

  Stmt *div(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::div, load(op1), load(op2));
  }

  Stmt *sel(Stmt *op1, Stmt *op2, Stmt *op3) {
    return insert<TernaryOpStmt>(TernaryOpType::select, load(op1), load(op2),
                                 load(op3));
  }

  Stmt *cos(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::cos, load(op1));
  }

  Stmt *sin(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::sin, load(op1));
  }

  Stmt *log(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::log, load(op1));
  }

  Stmt *pow(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::pow, load(op1), load(op2));
  }

 public:
  virtual Stmt *insert_grad_stmt(std::unique_ptr<Stmt> &&stmt) = 0;

  template <typename T, typename... Args>
  Stmt *insert(Args &&...args) {
    return insert_grad_stmt(Stmt::make<T>(args...));
  }

  template <typename T>
  Stmt *insert_const_for_grad(const DataType &dtype, Stmt *stmt, const T &val) {
    auto zero = insert<ConstStmt>(
        TypedConstant(dtype.ptr_removed().get_element_type(), val));
    if (dtype.ptr_removed()->is<TensorType>()) {
      auto t_dtype = dtype.ptr_removed()->as<TensorType>();
      std::vector<Stmt *> values(t_dtype->get_num_elements(), zero);
      zero = insert<MatrixInitStmt>(values);
      zero->ret_type = dtype.ptr_removed();
    }
    return zero;
  }

  void visit(AllocaStmt *alloca) override {
    // do nothing.
  }

  void visit(AdStackAllocaStmt *alloca) override {
    // do nothing.
  }

  void visit(ArgLoadStmt *stmt) override {
    // do nothing.
  }

  void visit(GetElementStmt *stmt) override {
    // do nothing
  }

  void visit(LoopIndexStmt *stmt) override {
    // do nothing.
  }

  void visit(MatrixPtrStmt *stmt) override {
    // do nothing.
  }

  void visit(PrintStmt *print_stmt) override {
    // do nothing
  }

  void visit(ConstStmt *const_stmt) override {
    // do nothing
  }

  void visit(ReturnStmt *stmt) override {
    // do nothing
  }

  void visit(WhileControlStmt *stmt) override {
    TI_NOT_IMPLEMENTED
  }

  void visit(ContinueStmt *stmt) override {
    TI_NOT_IMPLEMENTED;
  }

  void visit(WhileStmt *stmt) override {
    TI_NOT_IMPLEMENTED
  }

  void visit(GlobalPtrStmt *stmt) override {
    // do nothing
  }

  void visit(ExternalPtrStmt *stmt) override {
    // do nothing
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    // do nothing
  }

  void visit(DecorationStmt *stmt) override {
    // do nothing
  }

  Stmt *load(Stmt *alloc) {
    TI_ASSERT(alloc != nullptr);
    if (alloc->is<AllocaStmt>() || alloc->is<MatrixPtrStmt>()) {
      return insert<LocalLoadStmt>(alloc);
    } else {
      // non alloca
      return alloc;
    }
  }

  bool gradients_stopped(GlobalLoadStmt *stmt, SNode *snode) {
    for (auto block = stmt->parent; block; block = block->parent_block()) {
      for (auto s : block->stop_gradients) {
        if (s == snode) {
          return true;
        }
      }
    }
    return false;
  }

  void visit(AssertStmt *stmt) override {
    // do nothing
  }

  void visit(RangeAssumptionStmt *stmt) override {
    // do nothing
  }

  void visit(LinearizeStmt *stmt) override {
    // do nothing
  }

  void visit(IntegerOffsetStmt *stmt) override {
    // do nothing
  }

  void visit(RandStmt *stmt) override {
    TI_ERROR("RandStmt not supported in AutoDiff for now.");
  }
};

// Generate the adjoint version of an independent block
class MakeAdjoint : public ADTransform {
 public:
  using ADTransform::visit;
  Block *current_block;
  Block *alloca_block;
  // Backup the forward pass (the forward pass might be modified during the
  // MakeAdjoint) for search whether a GlobalLoadStmt is inside a for-loop when
  // allocating adjoint (see the function `adjoint`) Should be stored
  // 1. Before entering a for-loop body
  // 2. Before entering a if-stmt
  // Should be restored after processing every statement in the two cases above
  Block *forward_backup;
  std::map<Stmt *, Stmt *> adjoint_stmt;

  explicit MakeAdjoint(Block *block) {
    current_block = nullptr;
    alloca_block = block;
    forward_backup = block;
  }

  static void run(Block *block) {
    auto p = MakeAdjoint(block);
    block->accept(&p);
  }

  // TODO: current block might not be the right block to insert adjoint
  // instructions!
  void visit(Block *block) override {
    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : block->statements) {
      statements.push_back(stmt.get());
    }
    std::reverse(statements.begin(), statements.end());  // reverse-mode AD...
    for (auto stmt : statements) {
      current_block = block;
      stmt->accept(this);
    }
  }

  Stmt *insert_grad_stmt(std::unique_ptr<Stmt> &&stmt) override {
    auto ptr = stmt.get();
    current_block->insert(std::move(stmt), -1);
    return ptr;
  }

  // Accumulate [value] to the adjoint of [primal]
  void accumulate(Stmt *primal, Stmt *value) {
    auto alloca_ = adjoint(primal);
    if (!alloca_ || alloca_->is<ConstStmt>()) {
      return;  // primal may be int variable
    }
    if (alloca_->is<AdStackAllocaStmt>()) {
      auto alloca = alloca_->cast<AdStackAllocaStmt>();
      if (is_real(alloca->ret_type.get_element_type())) {
        insert<AdStackAccAdjointStmt>(alloca, load(value));
      }
    } else {
      TI_ASSERT(alloca_->is<AllocaStmt>());
      auto alloca = alloca_->as<AllocaStmt>();
      auto local_load = insert<LocalLoadStmt>(alloca);
      local_load->ret_type = alloca->ret_type.ptr_removed();
      insert<LocalStoreStmt>(alloca, add(local_load, value));
    }
  }

  Stmt *adjoint(Stmt *stmt) {
    DataType adjoint_dtype = stmt->ret_type.ptr_removed();
    if (adjoint_dtype->is<TensorType>()) {
      DataType prim_dtype = PrimitiveType::f32;
      if (is_real(adjoint_dtype.get_element_type())) {
        prim_dtype = adjoint_dtype.get_element_type();
      }
      adjoint_dtype = TypeFactory::get_instance().get_tensor_type(
          adjoint_dtype->as<TensorType>()->get_shape(), prim_dtype);
    } else if (stmt->is<MatrixPtrStmt>()) {
      // pass
    } else if (!is_real(adjoint_dtype) || stmt->is<ConstStmt>()) {
      return constant(0);
    }

    if (adjoint_stmt.find(stmt) == adjoint_stmt.end()) {
      // normal SSA cases

      // create the alloca
      // auto alloca =
      //    Stmt::make<AllocaStmt>(get_current_program().config.gradient_dt);
      // maybe it's better to use the statement data type than the default type
      auto alloca = Stmt::make<AllocaStmt>(adjoint_dtype);
      adjoint_stmt[stmt] = alloca.get();

      // We need to insert the alloca in the block of GlobalLoadStmt when the
      // GlobalLoadStmt is not inside a range-for
      // Code sample:
      // a and b require grad
      // Case 1 (GlobalLoadStmt is outside the for-loop, compute 5 times and
      // accumulate once, alloca history value is needed):
      // for i in range(5):
      //     p = a[i]
      //     q = b[i]
      //     for _ in range(5)
      //         q += p

      // Case 2 (GlobalLoadStmt is inside the for-loop, compute once and
      // accumulate immediately, alloca history value can be discarded):
      // for i in range(5):
      //     q = b[i]
      //     for _ in range(5)
      //         q += a[i]
      if (stmt->is<GlobalLoadStmt>() &&
          (stmt->parent->parent_stmt() != nullptr) &&
          stmt->parent->parent_stmt()->is<RangeForStmt>()) {
        // Check whether this GlobalLoadStmt is in the body of a for-loop by
        // searching in the backup forward pass If not (Case 1), the alloca
        // should not be clear every iteration, therefore, we need to insert the
        // alloca in the block of the GlobalLoadStmt i.e., where GlobalLoadStmt
        // is defined
        if (forward_backup->locate(stmt->as<GlobalLoadStmt>()) == -1) {
          stmt->as<GlobalLoadStmt>()->parent->insert(std::move(alloca), 0);
        } else {
          alloca_block->insert(std::move(alloca), 0);
        }
      } else {
        alloca_block->insert(std::move(alloca), 0);
      }
    }
    return adjoint_stmt[stmt];
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->op_type == UnaryOpType::floor ||
        stmt->op_type == UnaryOpType::ceil) {
      // do nothing
    } else if (stmt->op_type == UnaryOpType::neg) {
      accumulate(stmt->operand, negate(adjoint(stmt)));
    } else if (stmt->op_type == UnaryOpType::abs) {
      accumulate(stmt->operand, mul(adjoint(stmt), sgn(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::sin) {
      accumulate(stmt->operand, mul(adjoint(stmt), cos(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::cos) {
      accumulate(stmt->operand, negate(mul(adjoint(stmt), sin(stmt->operand))));
    } else if (stmt->op_type == UnaryOpType::tan) {
      // The derivative of `tan` is `1 / cos^2`, which has many singular points
      // causing NaNs. Though the NaNs are expected, it is error prone and hard
      // to debug. Therefore we currently don't support computing derivative for
      // `tan`.
      TI_NOT_IMPLEMENTED;
    } else if (stmt->op_type == UnaryOpType::tanh) {
      accumulate(
          stmt->operand,
          mul(adjoint(stmt), sub(constant(1, stmt->ret_type), sqr(stmt))));
    } else if (stmt->op_type == UnaryOpType::asin) {
      accumulate(
          stmt->operand,
          mul(adjoint(stmt),
              div(constant(1, stmt->ret_type),
                  sqrt(sub(constant(1, stmt->ret_type), sqr(stmt->operand))))));
    } else if (stmt->op_type == UnaryOpType::acos) {
      accumulate(
          stmt->operand,
          mul(adjoint(stmt), negate(div(constant(1, stmt->ret_type),
                                        sqrt(sub(constant(1, stmt->ret_type),
                                                 sqr(stmt->operand)))))));
    } else if (stmt->op_type == UnaryOpType::exp) {
      accumulate(stmt->operand, mul(adjoint(stmt), stmt));
    } else if (stmt->op_type == UnaryOpType::log) {
      accumulate(stmt->operand, div(adjoint(stmt), stmt->operand));
    } else if (stmt->op_type == UnaryOpType::sqrt) {
      accumulate(stmt->operand,
                 mul(adjoint(stmt),
                     div(constant(0.5f, stmt->ret_type), sqrt(stmt->operand))));
    } else if (stmt->op_type == UnaryOpType::rsqrt) {
      accumulate(stmt->operand,
                 mul(adjoint(stmt), mul(constant(-0.5f, stmt->ret_type),
                                        pow(rsqrt(stmt->operand),
                                            constant(3, stmt->ret_type)))));
    } else if (stmt->op_type == UnaryOpType::cast_value) {
      if (is_real(stmt->cast_type.get_element_type()) &&
          is_real(stmt->operand->ret_type.get_element_type())) {
        accumulate(stmt->operand, adjoint(stmt));
      }
    } else if (stmt->op_type == UnaryOpType::logic_not) {
      // do nothing
    } else {
      TI_P(unary_op_type_name(stmt->op_type));
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(BinaryOpStmt *bin) override {
    if (bin->op_type == BinaryOpType::add) {
      accumulate(bin->lhs, adjoint(bin));
      accumulate(bin->rhs, adjoint(bin));
    } else if (bin->op_type == BinaryOpType::sub) {
      accumulate(bin->lhs, adjoint(bin));
      accumulate(bin->rhs, negate(adjoint(bin)));
    } else if (bin->op_type == BinaryOpType::mul) {
      // d (x * y) = y * dx + x * dy
      accumulate(bin->lhs, mul(adjoint(bin), bin->rhs));
      accumulate(bin->rhs, mul(adjoint(bin), bin->lhs));
    } else if (bin->op_type == BinaryOpType::mod) {
      // Do nothing
    } else if (bin->op_type == BinaryOpType::div) {
      accumulate(bin->lhs, div(adjoint(bin), bin->rhs));
      accumulate(bin->rhs, negate(div(mul(adjoint(bin), bin->lhs),
                                      mul(bin->rhs, bin->rhs))));
    } else if (bin->op_type == BinaryOpType::atan2) {
      auto numerator = add(sqr(bin->lhs), sqr(bin->rhs));
      accumulate(bin->lhs, div(mul(adjoint(bin), bin->rhs), numerator));
      accumulate(bin->rhs, negate(div(mul(adjoint(bin), bin->lhs), numerator)));
    } else if (bin->op_type == BinaryOpType::pow) {
      // d (x ^ y) = x ^ (y-1) * (y * dx + log(x) * x * dy)
      auto common_coeff = pow(
          bin->lhs, sub(bin->rhs, constant(1, bin->ret_type)));  // x ^ (y-1)
      accumulate(bin->lhs, mul(adjoint(bin), mul(bin->rhs, common_coeff)));
      accumulate(bin->rhs, mul(adjoint(bin), mul(log(bin->lhs),
                                                 mul(bin->lhs, common_coeff))));
    } else if (bin->op_type == BinaryOpType::min ||
               bin->op_type == BinaryOpType::max) {
      auto cmp = bin->op_type == BinaryOpType::min ? cmp_lt(bin->lhs, bin->rhs)
                                                   : cmp_lt(bin->rhs, bin->lhs);
      auto zero = insert_const_for_grad(bin->ret_type, bin, 0);
      accumulate(bin->lhs, sel(cmp, adjoint(bin), zero));
      accumulate(bin->rhs, sel(cmp, zero, adjoint(bin)));
    } else if (bin->op_type == BinaryOpType::floordiv) {
      // do nothing
    } else if (is_comparison(bin->op_type) || is_bit_op(bin->op_type) ||
               binary_is_logical(bin->op_type)) {
      // do nothing

    } else {
      TI_WARN("gradient of binary op {}\n{}", binary_op_type_name(bin->op_type),
              bin->get_tb());
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    TI_ASSERT(stmt->op_type == TernaryOpType::select);
    auto zero = insert_const_for_grad(stmt->ret_type, stmt, 0);
    accumulate(stmt->op2,
               insert<TernaryOpStmt>(TernaryOpType::select, stmt->op1,
                                     load(adjoint(stmt)), zero));
    accumulate(stmt->op3,
               insert<TernaryOpStmt>(TernaryOpType::select, stmt->op1, zero,
                                     load(adjoint(stmt))));
  }

  void visit(IfStmt *if_stmt) override {
    auto new_if = Stmt::make_typed<IfStmt>(if_stmt->cond);
    if (if_stmt->true_statements) {
      new_if->set_true_statements(std::make_unique<Block>());
      auto old_current_block = current_block;
      // Backup forward pass
      forward_backup = if_stmt->true_statements.get();

      current_block = new_if->true_statements.get();
      for (int i = if_stmt->true_statements->statements.size() - 1; i >= 0;
           i--) {
        if_stmt->true_statements->statements[i]->accept(this);
        // Restore forward pass
        forward_backup = if_stmt->true_statements.get();
      }

      current_block = old_current_block;
    }
    if (if_stmt->false_statements) {
      new_if->set_false_statements(std::make_unique<Block>());
      auto old_current_block = current_block;

      // Backup forward pass
      forward_backup = if_stmt->false_statements.get();

      current_block = new_if->false_statements.get();
      for (int i = if_stmt->false_statements->statements.size() - 1; i >= 0;
           i--) {
        if_stmt->false_statements->statements[i]->accept(this);
        // Restore forward pass
        forward_backup = if_stmt->false_statements.get();
      }
      current_block = old_current_block;
    }
    insert_grad_stmt(std::move(new_if));
  }

  void visit(RangeForStmt *for_stmt) override {
    auto new_for = for_stmt->clone();
    auto new_for_ptr = new_for->as<RangeForStmt>();
    new_for_ptr->reversed = !new_for_ptr->reversed;
    insert_grad_stmt(std::move(new_for));
    const int len = new_for_ptr->body->size();

    for (int i = 0; i < len; i++) {
      new_for_ptr->body->erase(0);
    }

    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : for_stmt->body->statements) {
      statements.push_back(stmt.get());
    }
    std::reverse(statements.begin(), statements.end());  // reverse-mode AD...
    auto old_alloca_block = alloca_block;
    auto old_forward_backup =
        forward_backup;  // store the block which is not inside the current IB,
                         // such as outer most loop
    // Backup the forward pass
    forward_backup = for_stmt->body.get();
    for (auto stmt : statements) {
      alloca_block = new_for_ptr->body.get();
      current_block = new_for_ptr->body.get();
      stmt->accept(this);
      // Restore the forward pass
      forward_backup = for_stmt->body.get();
    }
    forward_backup = old_forward_backup;
    alloca_block = old_alloca_block;
  }

  void visit(StructForStmt *for_stmt) override {
    alloca_block = for_stmt->body.get();
    for_stmt->body->accept(this);
  }

  // Equivalent to AdStackLoadTopStmt when no stack is needed
  void visit(LocalLoadStmt *stmt) override {
    // TI_ASSERT(!needs_grad(stmt->ret_type));
    if (is_real(stmt->ret_type.get_element_type()))
      accumulate(stmt->src, load(adjoint(stmt)));
  }

  // Equivalent to AdStackPushStmt when no stack is needed
  void visit(LocalStoreStmt *stmt) override {
    accumulate(stmt->val, load(adjoint(stmt->dest)));

    // Clear the adjoint of the dest after local store,
    // Because LocalStoreStmt overwrites the dest,
    // 1. If the alloca is inside a loop, the adjoint of this alloca of this
    // iteration should be cleared after this iteration has been done
    // 2. If the alloca serves as the dest of multiple LocalStoreStmt, only the
    // last LocalStoreStmt should be taken account of
    auto dest_type = stmt->dest->ret_type.ptr_removed();
    if (is_real(dest_type.get_element_type())) {
      auto dtype = dest_type;
      auto zero = insert_const_for_grad(dtype, stmt, 0);
      insert<LocalStoreStmt>(adjoint(stmt->dest), zero);
    }
  }

  void visit(AdStackLoadTopStmt *stmt) override {
    if (is_real(stmt->ret_type.get_element_type()))
      insert<AdStackAccAdjointStmt>(stmt->stack, load(adjoint(stmt)));
  }

  void visit(AdStackPushStmt *stmt) override {
    accumulate(stmt->v, insert<AdStackLoadTopAdjStmt>(stmt->stack));
    insert<AdStackPopStmt>(stmt->stack);
  }

  void visit(GlobalLoadStmt *stmt) override {
    // issue global store to adjoint

    if (stmt->src->is<ExternalPtrStmt>() ||
        (stmt->src->is<MatrixPtrStmt>() &&
         stmt->src->as<MatrixPtrStmt>()->origin->is<ExternalPtrStmt>())) {
      ExternalPtrStmt *src = nullptr;
      bool is_ptr_offset = false;
      if (stmt->src->is<MatrixPtrStmt>()) {
        src = stmt->src->as<MatrixPtrStmt>()->origin->as<ExternalPtrStmt>();
        is_ptr_offset = true;
      } else {
        src = stmt->src->as<ExternalPtrStmt>();
      }
      auto arg = src->base_ptr->as<ArgLoadStmt>();
      if (arg->ret_type.ptr_removed()->as<StructType>()->elements().size() >
          TypeFactory::GRAD_PTR_POS_IN_NDARRAY) {
        TI_ASSERT_INFO(!src->is_grad,
                       "Cannot automatically differentiate through a grad "
                       "tensor, if you really want to do that, pass the grad "
                       "tensor into the kernel directly");
        auto adj_ptr =
            insert<ExternalPtrStmt>(src->base_ptr, src->indices, src->ndim,
                                    src->element_shape, /*is_grad=*/true);
        adj_ptr->ret_type = src->ret_type;

        if (is_ptr_offset) {
          adj_ptr = insert<MatrixPtrStmt>(
              adj_ptr, stmt->src->as<MatrixPtrStmt>()->offset);
          adj_ptr->ret_type = stmt->src->ret_type;
          adj_ptr->ret_type.set_is_pointer(true);
        }
        insert<AtomicOpStmt>(AtomicOpType::add, adj_ptr, load(adjoint(stmt)));
      }
      return;
    }

    if (stmt->src->is<GlobalPtrStmt>() ||
        (stmt->src->is<MatrixPtrStmt>() &&
         stmt->src->as<MatrixPtrStmt>()->origin->is<GlobalPtrStmt>())) {
      GlobalPtrStmt *src = nullptr;
      bool is_ptr_offset = false;
      if (stmt->src->is<MatrixPtrStmt>()) {
        is_ptr_offset = true;
        src = stmt->src->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
      } else {
        src = stmt->src->as<GlobalPtrStmt>();
      }

      auto snode = src->snode;
      if (!snode->has_adjoint()) {
        // No adjoint SNode. Do nothing
        return;
      }
      if (gradients_stopped(stmt, snode)) {
        // gradients stopped, do nothing.
        return;
      }
      TI_ASSERT(snode->get_adjoint() != nullptr);
      snode = snode->get_adjoint();
      auto adj_ptr = insert<GlobalPtrStmt>(snode, src->indices);
      adj_ptr->ret_type = src->ret_type;
      if (is_ptr_offset) {
        adj_ptr = insert<MatrixPtrStmt>(adj_ptr,
                                        stmt->src->as<MatrixPtrStmt>()->offset);
      }
      insert<AtomicOpStmt>(AtomicOpType::add, adj_ptr, load(adjoint(stmt)));
      return;
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    // erase and replace with global load adjoint

    Stmt *adjoint_ptr{nullptr};
    if (stmt->dest->is<ExternalPtrStmt>() ||
        (stmt->dest->is<MatrixPtrStmt>() &&
         stmt->dest->as<MatrixPtrStmt>()->origin->is<ExternalPtrStmt>())) {
      ExternalPtrStmt *dest = nullptr;
      bool is_ptr_offset = false;
      if (stmt->dest->is<MatrixPtrStmt>()) {
        is_ptr_offset = true;
        dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<ExternalPtrStmt>();
      } else {
        dest = stmt->dest->as<ExternalPtrStmt>();
      }

      auto arg = dest->base_ptr->as<ArgLoadStmt>();
      if (arg->ret_type.ptr_removed()->as<StructType>()->elements().size() <=
          TypeFactory::GRAD_PTR_POS_IN_NDARRAY) {
        return;
      }
      TI_ASSERT_INFO(!dest->is_grad,
                     "Cannot automatically differentiate through a grad "
                     "tensor, if you really want to do that, pass the grad "
                     "tensor into the kernel directly");
      adjoint_ptr = insert<ExternalPtrStmt>(dest->base_ptr, dest->indices,
                                            dest->ndim, dest->element_shape,
                                            /*is_grad=*/true);
      adjoint_ptr->ret_type = dest->ret_type;

      if (is_ptr_offset) {
        adjoint_ptr = insert<MatrixPtrStmt>(
            adjoint_ptr, stmt->dest->as<MatrixPtrStmt>()->offset);
        adjoint_ptr->ret_type = stmt->dest->ret_type;
        adjoint_ptr->ret_type.set_is_pointer(true);
      }

      accumulate(stmt->val, insert<GlobalLoadStmt>(adjoint_ptr));
    }

    if (stmt->dest->is<GlobalPtrStmt>() ||
        (stmt->dest->is<MatrixPtrStmt>() &&
         stmt->dest->as<MatrixPtrStmt>()->origin->is<GlobalPtrStmt>())) {
      GlobalPtrStmt *dest = nullptr;
      bool is_ptr_offset = false;
      if (stmt->dest->is<MatrixPtrStmt>()) {
        is_ptr_offset = true;
        dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
      } else {
        dest = stmt->dest->as<GlobalPtrStmt>();
      }

      auto snode = dest->snode;
      if (!snode->has_adjoint()) {
        // no gradient (likely integer types)
        return;
      }
      TI_ASSERT(snode->get_adjoint() != nullptr);
      snode = snode->get_adjoint();
      adjoint_ptr = insert<GlobalPtrStmt>(snode, dest->indices);
      adjoint_ptr->ret_type = dest->ret_type;
      if (is_ptr_offset) {
        adjoint_ptr = insert<MatrixPtrStmt>(
            adjoint_ptr, stmt->dest->as<MatrixPtrStmt>()->offset);
      }
      accumulate(stmt->val, insert<GlobalLoadStmt>(adjoint_ptr));
    }

    // Clear the gradient after accumulation finished.
    auto zero =
        insert_const_for_grad(adjoint_ptr->ret_type.ptr_removed(), stmt, 0);
    insert<GlobalStoreStmt>(adjoint_ptr, zero);

    stmt->parent->erase(stmt);
  }

  void visit(AtomicOpStmt *stmt) override {
    if (stmt->dest->is<ExternalPtrStmt>() ||
        (stmt->dest->is<MatrixPtrStmt>() &&
         stmt->dest->as<MatrixPtrStmt>()->origin->is<ExternalPtrStmt>())) {
      ExternalPtrStmt *dest = nullptr;
      bool is_ptr_offset = false;
      if (stmt->dest->is<MatrixPtrStmt>()) {
        is_ptr_offset = true;
        dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<ExternalPtrStmt>();
      } else {
        dest = stmt->dest->as<ExternalPtrStmt>();
      }

      auto arg = dest->base_ptr->as<ArgLoadStmt>();
      if (arg->ret_type.ptr_removed()->as<StructType>()->elements().size() >
          TypeFactory::GRAD_PTR_POS_IN_NDARRAY) {
        TI_ASSERT_INFO(!dest->is_grad,
                       "Cannot automatically differentiate through a grad "
                       "tensor, if you really want to do that, pass the grad "
                       "tensor into the kernel directly");
        auto adjoint_ptr =
            insert<ExternalPtrStmt>(dest->base_ptr, dest->indices, dest->ndim,
                                    dest->element_shape, /*is_grad=*/true);
        adjoint_ptr->ret_type = dest->ret_type;

        if (is_ptr_offset) {
          adjoint_ptr = insert<MatrixPtrStmt>(
              adjoint_ptr, stmt->dest->as<MatrixPtrStmt>()->offset);

          adjoint_ptr->ret_type = stmt->dest->ret_type;
          adjoint_ptr->ret_type.set_is_pointer(true);
        }
        adjoint_ptr->ret_type = dest->ret_type;
        accumulate(stmt->val, insert<GlobalLoadStmt>(adjoint_ptr));
        stmt->parent->erase(stmt);
      }
      return;
    }

    if (stmt->dest->is<GlobalPtrStmt>() ||
        (stmt->dest->is<MatrixPtrStmt>() &&
         stmt->dest->as<MatrixPtrStmt>()->origin->is<GlobalPtrStmt>())) {
      GlobalPtrStmt *dest = nullptr;
      bool is_ptr_offset = false;
      if (stmt->dest->is<MatrixPtrStmt>()) {
        is_ptr_offset = true;
        dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
      } else {
        dest = stmt->dest->as<GlobalPtrStmt>();
      }

      auto snode = dest->snode;
      if (!snode->has_adjoint()) {
        // no gradient (likely integer types)
        return;
      }

      TI_ASSERT(snode->get_adjoint() != nullptr);
      snode = snode->get_adjoint();
      auto adjoint_ptr = insert<GlobalPtrStmt>(snode, dest->indices);
      adjoint_ptr->ret_type = dest->ret_type;
      if (is_ptr_offset) {
        adjoint_ptr = insert<MatrixPtrStmt>(
            adjoint_ptr, stmt->dest->as<MatrixPtrStmt>()->offset);
      }
      accumulate(stmt->val, insert<GlobalLoadStmt>(adjoint_ptr));
      stmt->parent->erase(stmt);
      return;
    }
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<GlobalPtrStmt>() ||
        stmt->origin->is<ExternalPtrStmt>()) {
      /*
        The case of MatrixPtrStmt(GlobalPtrStmt, ...) is already handled in
        GlobalPtrStmt, GlobalStoreStmt and AtomicStmt

        TODO(zhanlue): Try to separate out the chain rule for MatrixPtrStmt from
        GlobalPtrStmt, GlobalStoreStmt and AtomicStmt and migrate the logics
        here.
      */
      return;
    }

    DataType prim_dtype = PrimitiveType::f32;
    if (is_real(stmt->ret_type.ptr_removed().get_element_type())) {
      prim_dtype = stmt->ret_type.ptr_removed().get_element_type();
    }

    Stmt *adjoint_value = nullptr;
    if (stmt->offset->is<ConstStmt>()) {
      /*
      [Static index]
      Fwd:
      $0 = alloca <4 x i32>
      $1 = matrix ptr $0, 2 // offset = 2

      Adjoint:
      $3 = matrix init [0, 0, $1_adj, 0] // adjoint_value

      accumulate($0_adj, $3)
      */
      int offset = stmt->offset->as<ConstStmt>()->val.val_int32();

      auto tensor_type = stmt->origin->ret_type.ptr_removed()->as<TensorType>();
      int num_elements = tensor_type->get_num_elements();

      auto zero = insert_const_for_grad(prim_dtype, stmt, 0);
      std::vector<Stmt *> values;
      for (int i = 0; i < num_elements; i++) {
        if (i == offset) {
          values.push_back(load(adjoint(stmt)));
        } else {
          values.push_back(zero);
        }
      }
      auto matrix_init_stmt = insert<MatrixInitStmt>(values);
      matrix_init_stmt->ret_type = tensor_type;

      adjoint_value = matrix_init_stmt;

    } else {
      /*
       [Dynamic index]
       Fwd:
       $0 = alloca <4 x i32>
       $1 = matrix ptr $0, $offset

       Adjoint:
       $3 = matrix init [0.0, 0.0, 0.0, 0.0]
       $4 = matrix init [$1_adj, $1_adj, $1_adj, $1_adj]

       $5 = matrix init [0, 1, 2, 3]
       $6 = matrix init [offset, offset, offset, offset]
       $7 = bin_eq $6, $5
       $8 = select $7, $4, $3 // adjoint_value

       accumulate($0_adj, $7)
      */
      auto tensor_type = stmt->origin->ret_type.ptr_removed()->as<TensorType>();
      auto tensor_shape = tensor_type->get_shape();
      int num_elements = tensor_type->get_num_elements();

      auto zero = insert_const_for_grad(prim_dtype, stmt, 0);
      auto stmt_adj = load(adjoint(stmt));

      std::vector<Stmt *> zero_values(num_elements, zero);
      std::vector<Stmt *> stmt_adj_values(num_elements, stmt_adj);
      std::vector<Stmt *> offset_values(num_elements, stmt->offset);
      std::vector<Stmt *> indices_values(num_elements);
      for (size_t i = 0; i < num_elements; i++) {
        indices_values[i] = insert<ConstStmt>(TypedConstant((int32)i));
      }

      auto zero_matrix_init_stmt = insert<MatrixInitStmt>(zero_values);
      zero_matrix_init_stmt->ret_type = tensor_type;
      auto stmt_adj_matrix_init_stmt = insert<MatrixInitStmt>(stmt_adj_values);
      stmt_adj_matrix_init_stmt->ret_type = tensor_type;

      auto index_tensor_type = TypeFactory::get_instance().get_tensor_type(
          tensor_shape, PrimitiveType::i32);
      auto indices_matrix_init_stmt = insert<MatrixInitStmt>(indices_values);
      indices_matrix_init_stmt->ret_type = index_tensor_type;

      auto offset_matrix_init_stmt = insert<MatrixInitStmt>(offset_values);
      offset_matrix_init_stmt->ret_type = index_tensor_type;
      auto cmp_tensor_type = TypeFactory::get_instance().get_tensor_type(
          tensor_shape, PrimitiveType::u1);
      auto bin_eq_stmt =
          insert<BinaryOpStmt>(BinaryOpType::cmp_eq, offset_matrix_init_stmt,
                               indices_matrix_init_stmt);
      bin_eq_stmt->ret_type = cmp_tensor_type;

      auto select_stmt = insert<TernaryOpStmt>(
          TernaryOpType::select, bin_eq_stmt, stmt_adj_matrix_init_stmt,
          zero_matrix_init_stmt);
      adjoint_value = select_stmt;
    }

    accumulate(stmt->origin, adjoint_value);
  }

  void visit(MatrixInitStmt *stmt) override {
    auto adjoint_ptr = adjoint(stmt);

    auto tensor_type = stmt->ret_type->as<TensorType>();
    int num_elements = tensor_type->get_num_elements();

    for (size_t i = 0; i < num_elements; i++) {
      auto const_i = insert_const_for_grad(PrimitiveType::i32, stmt, i);

      auto matrix_ptr_stmt_i = insert<MatrixPtrStmt>(adjoint_ptr, const_i);
      matrix_ptr_stmt_i->ret_type = tensor_type->get_element_type();
      matrix_ptr_stmt_i->ret_type.set_is_pointer(true);

      accumulate(stmt->values[i], load(matrix_ptr_stmt_i));
    }
  }
};

// Forward mode autodiff
class MakeDual : public ADTransform {
 public:
  using ADTransform::visit;
  Stmt *current_stmt;
  Block *current_block;
  Block *alloca_block;
  std::map<Stmt *, Stmt *> dual_stmt;

  explicit MakeDual(Block *block) {
    current_stmt = nullptr;
    alloca_block = block;
    current_block = block;
  }

  static void run(Block *block) {
    auto p = MakeDual(block);
    block->accept(&p);
  }

  Stmt *insert_grad_stmt(std::unique_ptr<Stmt> &&stmt) override {
    auto ptr = stmt.get();
    current_stmt = current_stmt->insert_after_me(std::move(stmt));
    return ptr;
  }

  void visit(Block *block) override {
    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : block->statements) {
      statements.push_back(stmt.get());
    }
    for (auto stmt : statements) {
      current_stmt = stmt;
      stmt->accept(this);
    }
  }

  // Accumulate [value] to the dual of [primal]
  void accumulate(Stmt *primal, Stmt *value) {
    auto alloca_ = dual(primal);
    if (!alloca_ || alloca_->is<ConstStmt>())
      return;  // primal may be int variable

    TI_ASSERT(alloca_->is<AllocaStmt>());
    auto alloca = alloca_->as<AllocaStmt>();
    auto local_load = insert<LocalLoadStmt>(alloca);
    insert<LocalStoreStmt>(alloca, add(local_load, value));
  }

  Stmt *dual(Stmt *stmt) {
    auto dual_type = stmt->ret_type.ptr_removed();
    if (!is_real(dual_type.get_element_type()) || stmt->is<ConstStmt>()) {
      return constant(0);
    }
    if (dual_stmt.find(stmt) == dual_stmt.end()) {
      // normal SSA cases

      // create the alloca
      // auto alloca =
      //    Stmt::make<AllocaStmt>(get_current_program().config.gradient_dt);
      // maybe it's better to use the statement data type than the default type
      auto alloca = Stmt::make<AllocaStmt>(dual_type);
      dual_stmt[stmt] = alloca.get();

      // TODO: check whether there are any edge cases for the alloca_block
      alloca_block->insert(std::move(alloca), 0);
    }
    return dual_stmt[stmt];
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->op_type == UnaryOpType::neg) {
      accumulate(stmt, negate(dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::abs) {
      accumulate(stmt, mul(sgn(stmt->operand), dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::sin) {
      accumulate(stmt, mul(cos(stmt->operand), dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::cos) {
      accumulate(stmt, negate(mul(sin(stmt->operand), dual(stmt->operand))));
    } else if (stmt->op_type == UnaryOpType::tan) {
      // The derivative of `tan` is `1 / cos^2`, which has many singular points
      // causing NaNs. Though the NaNs are expected, it is error prone and hard
      // to debug. Therefore we currently don't support computing derivative for
      // `tan`.
      TI_NOT_IMPLEMENTED;
    } else if (stmt->op_type == UnaryOpType::tanh) {
      accumulate(stmt, mul(sub(constant(1), sqr(stmt)), dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::asin) {
      accumulate(stmt, mul(div(constant(1),
                               sqrt(sub(constant(1), sqr(stmt->operand)))),
                           dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::acos) {
      accumulate(stmt,
                 mul(negate(div(constant(1),
                                sqrt(sub(constant(1), sqr(stmt->operand))))),
                     dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::exp) {
      accumulate(stmt, mul(stmt, dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::log) {
      accumulate(stmt, div(dual(stmt->operand), stmt->operand));
    } else if (stmt->op_type == UnaryOpType::sqrt) {
      accumulate(stmt, mul(div(constant(0.5f), sqrt(stmt->operand)),
                           dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::rsqrt) {
      accumulate(stmt, mul(mul(constant(-0.5f),
                               pow(rsqrt(stmt->operand), constant(3))),
                           dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::cast_value) {
      if (is_real(stmt->cast_type.get_element_type()) &&
          is_real(stmt->operand->ret_type.get_element_type())) {
        accumulate(stmt, dual(stmt->operand));
      }
    } else if (stmt->op_type == UnaryOpType::logic_not) {
      // do nothing
    } else {
      TI_P(unary_op_type_name(stmt->op_type));
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(BinaryOpStmt *bin) override {
    if (bin->op_type == BinaryOpType::add) {
      accumulate(bin, dual(bin->lhs));
      accumulate(bin, dual(bin->rhs));
    } else if (bin->op_type == BinaryOpType::sub) {
      accumulate(bin, dual(bin->lhs));
      accumulate(bin, negate(dual(bin->rhs)));
    } else if (bin->op_type == BinaryOpType::mul) {
      // d (x * y) = y * dx + x * dy
      accumulate(bin, mul(bin->lhs, dual(bin->rhs)));
      accumulate(bin, mul(bin->rhs, dual(bin->lhs)));
    } else if (bin->op_type == BinaryOpType::mod) {
      // Do nothing
    } else if (bin->op_type == BinaryOpType::div) {
      accumulate(bin, div(dual(bin->lhs), bin->rhs));
      accumulate(bin, negate(div(mul(dual(bin->rhs), bin->lhs),
                                 mul(bin->rhs, bin->rhs))));
    } else if (bin->op_type == BinaryOpType::atan2) {
      auto numerator = add(sqr(bin->lhs), sqr(bin->rhs));
      accumulate(bin, div(mul(bin->rhs, dual(bin->lhs)), numerator));
      accumulate(bin, negate(div(mul(bin->lhs, dual(bin->rhs)), numerator)));
    } else if (bin->op_type == BinaryOpType::pow) {
      // d (x ^ y) = x ^ (y-1) * (y * dx + log(x) * x * dy)
      auto common_coeff =
          pow(bin->lhs, sub(bin->rhs, constant(1)));  // x ^ (y-1)
      accumulate(bin, mul(dual(bin->lhs), mul(bin->rhs, common_coeff)));
      accumulate(bin, mul(dual(bin->rhs),
                          mul(log(bin->lhs), mul(bin->lhs, common_coeff))));
    } else if (bin->op_type == BinaryOpType::min ||
               bin->op_type == BinaryOpType::max) {
      auto cmp = bin->op_type == BinaryOpType::min ? cmp_lt(bin->lhs, bin->rhs)
                                                   : cmp_lt(bin->rhs, bin->lhs);
      auto zero = insert_const_for_grad(bin->ret_type, bin, 0);
      accumulate(bin, sel(cmp, dual(bin->lhs), zero));
      accumulate(bin, sel(cmp, zero, dual(bin->rhs)));
    } else if (bin->op_type == BinaryOpType::floordiv) {
      // do nothing
    } else if (is_comparison(bin->op_type) || is_bit_op(bin->op_type)) {
      // do nothing
    } else {
      TI_WARN("gradient of binary op {}\n{}", binary_op_type_name(bin->op_type),
              bin->get_tb());
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    TI_ASSERT(stmt->op_type == TernaryOpType::select);
    auto zero = insert_const_for_grad(stmt->ret_type, stmt, 0);
    accumulate(stmt, insert<TernaryOpStmt>(TernaryOpType::select, stmt->op1,
                                           load(dual(stmt->op2)), zero));
    accumulate(stmt, insert<TernaryOpStmt>(TernaryOpType::select, stmt->op1,
                                           zero, load(dual(stmt->op3))));
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements) {
      std::vector<Stmt *> true_statements;
      for (auto &stmt : if_stmt->true_statements->statements) {
        true_statements.push_back(stmt.get());
      }

      for (auto stmt : true_statements) {
        current_stmt = stmt;
        stmt->accept(this);
      }
    }
    if (if_stmt->false_statements) {
      std::vector<Stmt *> false_statements;
      for (auto &stmt : if_stmt->false_statements->statements) {
        false_statements.push_back(stmt.get());
      }

      for (auto stmt : false_statements) {
        current_stmt = stmt;
        stmt->accept(this);
      }
    }
  }

  void visit(RangeForStmt *for_stmt) override {
    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : for_stmt->body->statements) {
      statements.push_back(stmt.get());
    }
    auto previous_alloca_block = alloca_block;
    alloca_block = for_stmt->body.get();
    for (auto stmt : statements) {
      current_stmt = stmt;
      stmt->accept(this);
    }
    alloca_block = previous_alloca_block;
  }

  void visit(StructForStmt *for_stmt) override {
    alloca_block = for_stmt->body.get();
    for_stmt->body->accept(this);
  }

  void visit(LocalLoadStmt *stmt) override {
    // TI_ASSERT(!needs_grad(stmt->ret_type));
    accumulate(stmt, dual(stmt->src));
  }

  void visit(LocalStoreStmt *stmt) override {
    // Clear the dual of the dest before local store,
    // Because LocalStoreStmt overwrites the dest,
    // If the alloca serves as the dest of multiple LocalStoreStmt, only the
    // last LocalStoreStmt should be taken account of, i.e, its history should
    // be cleared
    auto dtype = stmt->dest->ret_type.ptr_removed();
    if (is_real(dtype.get_element_type())) {
      auto zero = insert_const_for_grad(dtype, stmt, 0);
      insert<LocalStoreStmt>(dual(stmt->dest), zero);
    }

    accumulate(stmt->dest, dual(stmt->val));
  }

  void visit(GlobalLoadStmt *stmt) override {
    // issue global store to dual
    GlobalPtrStmt *src = nullptr;
    bool is_ptr_offset = false;
    if (stmt->src->is<MatrixPtrStmt>()) {
      is_ptr_offset = true;
      src = stmt->src->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    } else {
      src = stmt->src->as<GlobalPtrStmt>();
    }
    auto snode = src->snode;
    if (!snode->has_dual()) {
      // No dual SNode. Do nothing
      return;
    }
    if (gradients_stopped(stmt, snode)) {
      // gradients stopped, do nothing.
      return;
    }
    TI_ASSERT(snode->get_dual() != nullptr);
    snode = snode->get_dual();
    auto dual_ptr = insert<GlobalPtrStmt>(snode, src->indices);
    dual_ptr->ret_type = src->ret_type;
    if (is_ptr_offset) {
      dual_ptr = insert<MatrixPtrStmt>(dual_ptr,
                                       stmt->src->as<MatrixPtrStmt>()->offset);
    }
    accumulate(stmt, insert<GlobalLoadStmt>(dual_ptr));
  }

  void visit(GlobalStoreStmt *stmt) override {
    GlobalPtrStmt *dest = nullptr;
    bool is_ptr_offset = false;
    if (stmt->dest->is<MatrixPtrStmt>()) {
      is_ptr_offset = true;
      dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    } else {
      dest = stmt->dest->as<GlobalPtrStmt>();
    }
    auto snode = dest->snode;
    if (!snode->has_dual()) {
      // no gradient (likely integer types)
      return;
    }
    TI_ASSERT(snode->get_dual() != nullptr);
    snode = snode->get_dual();
    auto dual_ptr = insert<GlobalPtrStmt>(snode, dest->indices);
    dual_ptr->ret_type = dest->ret_type;
    if (is_ptr_offset) {
      dual_ptr = insert<MatrixPtrStmt>(dual_ptr,
                                       stmt->dest->as<MatrixPtrStmt>()->offset);
    }
    insert<AtomicOpStmt>(AtomicOpType::add, dual_ptr, load(dual(stmt->val)));
  }

  void visit(AtomicOpStmt *stmt) override {
    GlobalPtrStmt *dest = nullptr;
    bool is_ptr_offset = false;
    if (stmt->dest->is<MatrixPtrStmt>()) {
      is_ptr_offset = true;
      dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    } else {
      dest = stmt->dest->as<GlobalPtrStmt>();
    }
    auto snode = dest->snode;
    if (!snode->has_dual()) {
      // no gradient (likely integer types)
      return;
    }
    TI_ASSERT(snode->get_dual() != nullptr);
    snode = snode->get_dual();
    auto dual_ptr = insert<GlobalPtrStmt>(snode, dest->indices);
    dual_ptr->ret_type = dest->ret_type;
    if (is_ptr_offset) {
      dual_ptr = insert<MatrixPtrStmt>(dual_ptr,
                                       stmt->dest->as<MatrixPtrStmt>()->offset);
    }
    insert<AtomicOpStmt>(AtomicOpType::add, dual_ptr, load(dual(stmt->val)));
  }

  void visit(MatrixInitStmt *stmt) override {
    std::vector<Stmt *> duals;
    for (auto &s : stmt->values) {
      duals.push_back(dual(s));
    }
    auto dual_stmt = insert<MatrixInitStmt>(duals);
    dual_stmt->ret_type = stmt->ret_type;

    accumulate(stmt, dual_stmt);
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<GlobalPtrStmt>()) {
      // Handled in GlobalLoadStmt and GlobalStoreStmt
      return;
    }

    auto origin_dual = dual(stmt->origin);
    auto origin_dual_ptr = insert<MatrixPtrStmt>(origin_dual, stmt->offset);
    origin_dual_ptr->ret_type = stmt->ret_type;

    accumulate(stmt, origin_dual_ptr);
  }
};

class BackupSSA : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  Block *independent_block;
  std::map<Stmt *, Stmt *> backup_alloca;

  explicit BackupSSA(Block *independent_block)
      : independent_block(independent_block) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  Stmt *load(Stmt *stmt) {
    if (backup_alloca.find(stmt) == backup_alloca.end()) {
      auto alloca = Stmt::make<AllocaStmt>(stmt->ret_type.ptr_removed());
      auto alloca_ptr = alloca.get();
      independent_block->insert(std::move(alloca), 0);
      auto local_store = Stmt::make<LocalStoreStmt>(alloca_ptr, stmt);
      stmt->insert_after_me(std::move(local_store));
      backup_alloca[stmt] = alloca_ptr;
    }
    return backup_alloca[stmt];
  }

  void generic_visit(Stmt *stmt) {
    std::vector<Block *> leaf_to_root;
    auto t = stmt->parent;
    while (t != nullptr) {
      leaf_to_root.push_back(t);
      t = t->parent_block();
    }
    int num_operands = stmt->get_operands().size();
    for (int i = 0; i < num_operands; i++) {
      auto op = stmt->operand(i);
      if (op == nullptr) {
        continue;
      }
      if (std::find(leaf_to_root.begin(), leaf_to_root.end(), op->parent) ==
              leaf_to_root.end() &&
          !op->is<AllocaStmt>()) {
        if (op->is<AdStackLoadTopStmt>()) {
          // Just create another AdStackLoadTopStmt
          stmt->set_operand(i, stmt->insert_before_me(op->clone()));
        } else if (op->is<AdStackAllocaStmt>()) {
          // Backup AdStackAllocaStmt because it should not be local stored and
          // local loaded
          auto stack_alloca = op->as<AdStackAllocaStmt>();
          if (backup_alloca.find(op) == backup_alloca.end()) {
            auto backup_stack_alloca = Stmt::make<AdStackAllocaStmt>(
                stack_alloca->dt, stack_alloca->max_size);
            auto backup_stack_alloca_ptr = backup_stack_alloca.get();
            independent_block->insert(std::move(backup_stack_alloca), 0);
            backup_alloca[op] = backup_stack_alloca_ptr;
            // Replace usages of all blocks i.e., the entry point for the
            // replace is the top level block
            irpass::replace_all_usages_with(leaf_to_root.back(), op,
                                            backup_stack_alloca_ptr);
            // Erase the outdated AdStackAllocaStmt
            op->parent->erase(op);
          }
        } else if (op->is<ArgLoadStmt>()) {
          stmt->set_operand(i, stmt->insert_before_me(op->clone()));
        } else {
          auto alloca = load(op);
          stmt->set_operand(
              i, stmt->insert_before_me(Stmt::make<LocalLoadStmt>(alloca)));
        }
      }
    }
  }

  void visit(Stmt *stmt) override {
    generic_visit(stmt);
  }

  void visit(IfStmt *stmt) override {
    generic_visit(stmt);
    BasicStmtVisitor::visit(stmt);
  }

  // TODO: test operands for statements
  void visit(RangeForStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(StructForStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) override {
    TI_ERROR("WhileStmt not supported in AutoDiff for now.");
  }

  void visit(Block *block) override {
    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : block->statements) {
      statements.push_back(stmt.get());
    }
    for (auto stmt : statements) {
      TI_ASSERT(!stmt->erased);
      stmt->accept(this);
    }
  }

 public:
  static void run(Block *block) {
    BackupSSA pass(block);
    block->accept(&pass);
  }
};

namespace irpass {

// clang-format off
/*
Support for TensorType: How to handle MatrixPtrStmt & MatrixInitStmt

[Original Taichi Code]

@ti.kernel
def test(...):
    b = ti.Vector([0, 1, 2, 3])
    b[2] = 100
    y = b[3] * b[2] * x


[Forward]                          [Forward-Replaced]              [Backward]
$b = alloca Tensor<4 x i32>   -->  $b = adstack alloca <4 x i32>
$1 = matrix init [0, 1, 2, 3] -->  $1 = matrix init [0, 1, 2, 3]
                                                                       adstack pop
$2:  local store $b, $1       -->  adstack push $1                 --> acc($1_adj, adstack top adj())

$3 = matrix ptr $b, 2         -->  $2 = adstack top(is_ptr=True)   --> adstack acc adj($2_adj)

                                                                       acc($2_adj, $14)
                                   $3 = matrix ptr $2, 0           --> $14 = matrix_init({$3_adj, 0, 0, 0})

                                                                       acc($2_adj, $13)
                                   $4 = matrix ptr $2, 2           --> $13 = matrix_init({0, 0, $4_adj, 0})

                                                                       acc($2_adj, $12)
                                   $5 = matrix ptr $2, 3           --> $12 = matrix_init({0, 0, 0, $5_adj})

                                   $6 = load($3)                   --> acc($3_adj, $6_adj)
                                   $7 = load($4)                   --> acc($4_adj, $7_adj)
                                   $8 = load($5)                   --> acc($5_adj, $8_adj)

                                                                       acc($8_adj, matrix ptr($9_adj, 3))
                                                                       acc($7_adj, matrix ptr($9_adj, 2))
                                                                       acc(100_adj, matrix ptr($9_adj, 1))
                                   $9 = matrix_init($6,100,$7,$8)  --> acc($6_adj, matrix ptr($9_adj, 0))

                                                                       adstack pop
$4 = local store $3, 100      -->  adstack push $9                 --> acc($9_adj, adstack top adj())

                                   $10 = adstack top(is_ptr=True)  --> adstack acc adj($10_adj)

                                                                       acc($10_adj, $18)
$5 = matrix ptr $b, 3         -->  $11 = matrix ptr $10, 3         --> $18 = matrix_init({0, 0, 0, $11_adj})
$b3 = local load $5           -->  $b3 = local load $11            --> acc($11_adj, $b3_adj)

                                   $12 = adstack top(is_ptr=True)  --> adstack acc adj($12_adj)

                                                                       acc($12_adj, $17)
$6 = matrix ptr $b, 2         -->  $13 = matrix ptr $12, 2         --> $17 = matrix_init({0, 0, $13_adj, 0})
$b2 = local load $6           -->  $b2 = local load $13            --> acc($13_adj, $b2_adj)

                                                                       acc($b3_adj, $15)
                                                                       acc($b2_adj, $16)
                                                                       $16 = mul($tmp_adj, $b3)
$tmp = mul b3, b2             -->  $tmp = mul $b3, $b2             --> $15 = mul($tmp_adj, $b2)

                                                                       acc($tmp_adj, $14)
$y = mul $tmp, $x             -->  $y = mul $tmp, $x               --> $14 = mul($y_adj, $x)
*/
// clang-format on
void auto_diff(IRNode *root,
               const CompileConfig &config,
               AutodiffMode autodiff_mode,
               bool use_stack) {
  TI_AUTO_PROF;
  if (autodiff_mode == AutodiffMode::kReverse) {
    RegulateTensorTypedStatements::run(root);
    if (use_stack) {
      auto IB = IdentifyIndependentBlocks::run(root);
      ReverseOuterLoops::run(root, IB);

      for (auto ib : IB) {
        PromoteSSA2LocalVar::run(ib);
        ReplaceLocalVarWithStacks replace(config.ad_stack_size);
        ib->accept(&replace);
        type_check(root, config);

        MakeAdjoint::run(ib);
        type_check(root, config);
        BackupSSA::run(ib);
        irpass::analysis::verify(root);
      }
    } else {
      auto IB = IdentifyIndependentBlocks::run(root);
      ReverseOuterLoops::run(root, IB);
      type_check(root, config);
      for (auto ib : IB) {
        MakeAdjoint::run(ib);
        type_check(root, config);
        BackupSSA::run(ib);
        irpass::analysis::verify(root);
      }
    }
  } else if (autodiff_mode == AutodiffMode::kForward) {
    // Forward mode autodiff
    Block *block = root->as<Block>();
    MakeDual::run(block);
  }
  type_check(root, config);
  irpass::analysis::verify(root);
}

class GloablDataAccessRuleChecker : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void visit(GlobalLoadStmt *stmt) override {
    GlobalPtrStmt *src = nullptr;
    if (stmt->src->is<GlobalPtrStmt>()) {
      src = stmt->src->as<GlobalPtrStmt>();
    } else {
      TI_ASSERT(stmt->src->is<MatrixPtrStmt>());
      src = stmt->src->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    }
    auto snode = src->snode;
    if (!snode->has_adjoint_checkbit()) {
      return;
    }
    TI_ASSERT(snode->get_adjoint_checkbit() != nullptr);
    snode = snode->get_adjoint_checkbit();
    auto global_ptr =
        stmt->insert_after_me(Stmt::make<GlobalPtrStmt>(snode, src->indices));
    auto dtype = global_ptr->ret_type.ptr_removed();

    auto one = insert_const(dtype, global_ptr, 1, false /*insert_before_me*/);
    one->insert_after_me(Stmt::make<GlobalStoreStmt>(global_ptr, one));
  }

  void visit_gloabl_store_stmt_and_atomic_add(Stmt *stmt, GlobalPtrStmt *dest) {
    auto snode = dest->snode;
    if (!snode->has_adjoint_checkbit()) {
      return;
    }
    TI_ASSERT(snode->get_adjoint_checkbit() != nullptr);
    snode = snode->get_adjoint_checkbit();
    auto global_ptr =
        stmt->insert_before_me(Stmt::make<GlobalPtrStmt>(snode, dest->indices));
    auto global_load =
        stmt->insert_before_me(Stmt::make<GlobalLoadStmt>(global_ptr));
    auto dtype = global_ptr->ret_type.ptr_removed();
    auto zero = insert_const(dtype, stmt, 0, /*insert_before_me=*/true);
    auto check_equal = stmt->insert_before_me(
        Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_eq, global_load, zero));
    std::string msg = fmt::format(
        "(kernel={}) Breaks the global data access rule. Snode {} is "
        "overwritten unexpectedly.",
        kernel_name_, dest->snode->get_node_type_name());
    msg += "\n" + stmt->get_tb();

    stmt->insert_before_me(
        Stmt::make<AssertStmt>(check_equal, msg, std::vector<Stmt *>()));
  }

  void visit(GlobalStoreStmt *stmt) override {
    GlobalPtrStmt *dest = nullptr;
    if (stmt->dest->is<GlobalPtrStmt>()) {
      dest = stmt->dest->as<GlobalPtrStmt>();
    } else {
      TI_ASSERT(stmt->dest->is<MatrixPtrStmt>());
      dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    }
    visit_gloabl_store_stmt_and_atomic_add(stmt, dest);
  }

  void visit(AtomicOpStmt *stmt) override {
    GlobalPtrStmt *dest = nullptr;
    if (stmt->dest->is<GlobalPtrStmt>()) {
      dest = stmt->dest->as<GlobalPtrStmt>();
    } else {
      TI_ASSERT(stmt->dest->is<MatrixPtrStmt>());
      dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    }
    visit_gloabl_store_stmt_and_atomic_add(stmt, dest);
  }

  static void run(IRNode *root, const std::string &kernel_name) {
    GloablDataAccessRuleChecker checker;
    checker.kernel_name_ = kernel_name;
    root->accept(&checker);
  }

 private:
  std::string kernel_name_;
};

void differentiation_validation_check(IRNode *root,
                                      const CompileConfig &config,
                                      const std::string &kernel_name) {
  return irpass::GloablDataAccessRuleChecker::run(root, kernel_name);
}

}  // namespace irpass

}  // namespace taichi::lang
