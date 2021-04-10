#pragma once

#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

class IRBuilder {
 public:
  struct InsertPoint {
    Block *block{nullptr};
    int position{0};
  };

  IRBuilder();

  // Clear the IR and the insertion point.
  void reset();

  // Extract the IR.
  std::unique_ptr<IRNode> extract_ir();

  // General inserter. Returns stmt.get().
  template <typename XStmt>
  XStmt *insert(std::unique_ptr<XStmt> &&stmt) {
    return insert(std::move(stmt), &insert_point_);
  }

  // Insert to a specific insertion point.
  template <typename XStmt>
  static XStmt *insert(std::unique_ptr<XStmt> &&stmt,
                       InsertPoint *insert_point) {
    return insert_point->block
        ->insert(std::move(stmt), insert_point->position++)
        ->template as<XStmt>();
  }

  void set_insertion_point(InsertPoint new_insert_point);
  void set_insertion_point_to_after(Stmt *stmt);
  void set_insertion_point_to_before(Stmt *stmt);
  void set_insertion_point_to_true_branch(IfStmt *if_stmt);
  void set_insertion_point_to_false_branch(IfStmt *if_stmt);
  template <typename XStmt>
  void set_insertion_point_to_loop_begin(XStmt *loop) {
    using DecayedType = typename std::decay_t<XStmt>;
    if constexpr (!std::is_base_of_v<Stmt, DecayedType>) {
      TI_ERROR("The argument is not a statement.");
    }
    if constexpr (std::is_same_v<DecayedType, RangeForStmt> ||
                  std::is_same_v<DecayedType, StructForStmt> ||
                  std::is_same_v<DecayedType, WhileStmt>) {
      set_insertion_point({loop->body.get(), 0});
    } else {
      TI_ERROR("Statement {} is not a loop.", loop->name());
    }
  }

  // RAII handles insertion points automatically.
  class LoopGuard {
   public:
    // Set the insertion point to the beginning of the loop body.
    template <typename XStmt>
    explicit LoopGuard(IRBuilder &builder, XStmt *loop)
        : builder_(builder), loop_(loop) {
      location_ = (int)loop->parent->size() - 1;
      builder_.set_insertion_point_to_loop_begin(loop);
    }

    // Set the insertion point to the point after the loop.
    ~LoopGuard();

   private:
    IRBuilder &builder_;
    Stmt *loop_;
    int location_;
  };
  class IfGuard {
   public:
    // Set the insertion point to the beginning of the true/false branch.
    explicit IfGuard(IRBuilder &builder, IfStmt *if_stmt, bool true_branch);

    // Set the insertion point to the point after the if statement.
    ~IfGuard();

   private:
    IRBuilder &builder_;
    IfStmt *if_stmt_;
    int location_;
  };

  template <typename XStmt>
  LoopGuard get_loop_guard(XStmt *loop) {
    return LoopGuard(*this, loop);
  }
  IfGuard get_if_guard(IfStmt *if_stmt, bool true_branch) {
    return IfGuard(*this, if_stmt, true_branch);
  }

  // Control flows.
  RangeForStmt *create_range_for(Stmt *begin,
                                 Stmt *end,
                                 int vectorize = -1,
                                 int bit_vectorize = -1,
                                 int num_cpu_threads = 0,
                                 int block_dim = 0,
                                 bool strictly_serialized = false);
  StructForStmt *create_struct_for(SNode *snode,
                                   int vectorize = -1,
                                   int bit_vectorize = -1,
                                   int num_cpu_threads = 0,
                                   int block_dim = 0);
  WhileStmt *create_while_true();
  IfStmt *create_if(Stmt *cond);
  WhileControlStmt *create_break();
  ContinueStmt *create_continue();

  // Loop index.
  LoopIndexStmt *get_loop_index(Stmt *loop, int index = 0);

  // Constants. TODO: add more types
  ConstStmt *get_int32(int32 value);
  ConstStmt *get_int64(int64 value);
  ConstStmt *get_float32(float32 value);
  ConstStmt *get_float64(float64 value);

  // Load kernel arguments.
  ArgLoadStmt *create_arg_load(int arg_id, DataType dt, bool is_ptr);

  // The return value of the kernel.
  KernelReturnStmt *create_return(Stmt *value);

  // Unary operations. Returns the result.
  UnaryOpStmt *create_cast(Stmt *value, DataType output_type);  // cast by value
  UnaryOpStmt *create_bit_cast(Stmt *value, DataType output_type);
  UnaryOpStmt *create_neg(Stmt *value);
  UnaryOpStmt *create_not(Stmt *value);  // bitwise
  UnaryOpStmt *create_logical_not(Stmt *value);
  UnaryOpStmt *create_floor(Stmt *value);
  UnaryOpStmt *create_ceil(Stmt *value);
  UnaryOpStmt *create_abs(Stmt *value);
  UnaryOpStmt *create_sgn(Stmt *value);
  UnaryOpStmt *create_sqrt(Stmt *value);
  UnaryOpStmt *create_rsqrt(Stmt *value);
  UnaryOpStmt *create_sin(Stmt *value);
  UnaryOpStmt *create_asin(Stmt *value);
  UnaryOpStmt *create_cos(Stmt *value);
  UnaryOpStmt *create_acos(Stmt *value);
  UnaryOpStmt *create_tan(Stmt *value);
  UnaryOpStmt *create_tanh(Stmt *value);
  UnaryOpStmt *create_exp(Stmt *value);
  UnaryOpStmt *create_log(Stmt *value);

  // Binary operations. Returns the result.
  BinaryOpStmt *create_add(Stmt *l, Stmt *r);
  BinaryOpStmt *create_sub(Stmt *l, Stmt *r);
  BinaryOpStmt *create_mul(Stmt *l, Stmt *r);
  // l / r in C++
  BinaryOpStmt *create_div(Stmt *l, Stmt *r);
  // floor(1.0 * l / r) in C++
  BinaryOpStmt *create_floordiv(Stmt *l, Stmt *r);
  // 1.0 * l / r in C++
  BinaryOpStmt *create_truediv(Stmt *l, Stmt *r);
  BinaryOpStmt *create_mod(Stmt *l, Stmt *r);
  BinaryOpStmt *create_max(Stmt *l, Stmt *r);
  BinaryOpStmt *create_min(Stmt *l, Stmt *r);
  BinaryOpStmt *create_atan2(Stmt *l, Stmt *r);
  BinaryOpStmt *create_pow(Stmt *l, Stmt *r);
  // Bitwise operations. TODO: add logical operations when we support them
  BinaryOpStmt *create_and(Stmt *l, Stmt *r);
  BinaryOpStmt *create_or(Stmt *l, Stmt *r);
  BinaryOpStmt *create_xor(Stmt *l, Stmt *r);
  BinaryOpStmt *create_shl(Stmt *l, Stmt *r);
  BinaryOpStmt *create_shr(Stmt *l, Stmt *r);
  BinaryOpStmt *create_sar(Stmt *l, Stmt *r);
  // Comparisons.
  BinaryOpStmt *create_cmp_lt(Stmt *l, Stmt *r);
  BinaryOpStmt *create_cmp_le(Stmt *l, Stmt *r);
  BinaryOpStmt *create_cmp_gt(Stmt *l, Stmt *r);
  BinaryOpStmt *create_cmp_ge(Stmt *l, Stmt *r);
  BinaryOpStmt *create_cmp_eq(Stmt *l, Stmt *r);
  BinaryOpStmt *create_cmp_ne(Stmt *l, Stmt *r);

  // Ternary operations. Returns the result.
  TernaryOpStmt *create_select(Stmt *cond,
                               Stmt *true_result,
                               Stmt *false_result);

  // Print values and strings. Arguments can be Stmt* or std::string.
  template <typename... Args>
  PrintStmt *create_print(Args &&... args) {
    return insert(Stmt::make_typed<PrintStmt>(std::forward<Args>(args)...));
  }

  // Local variables.
  AllocaStmt *create_local_var(DataType dt);
  LocalLoadStmt *create_local_load(AllocaStmt *ptr);
  void create_local_store(AllocaStmt *ptr, Stmt *data);

  // Global variables.
  GlobalPtrStmt *create_global_ptr(SNode *snode,
                                   const std::vector<Stmt *> &indices);
  ExternalPtrStmt *create_external_ptr(ArgLoadStmt *ptr,
                                       const std::vector<Stmt *> &indices);
  template <typename XStmt>
  GlobalLoadStmt *create_global_load(XStmt *ptr) {
    using DecayedType = typename std::decay_t<XStmt>;
    if constexpr (!std::is_base_of_v<Stmt, DecayedType>) {
      TI_ERROR("The argument is not a statement.");
    }
    if constexpr (std::is_same_v<DecayedType, GlobalPtrStmt> ||
                  std::is_same_v<DecayedType, ExternalPtrStmt>) {
      return insert(Stmt::make_typed<GlobalLoadStmt>(ptr));
    } else {
      TI_ERROR("Statement {} is not a global pointer.", ptr->name());
    }
  }
  template <typename XStmt>
  void create_global_store(XStmt *ptr, Stmt *data) {
    using DecayedType = typename std::decay_t<XStmt>;
    if constexpr (!std::is_base_of_v<Stmt, DecayedType>) {
      TI_ERROR("The argument is not a statement.");
    }
    if constexpr (std::is_same_v<DecayedType, GlobalPtrStmt> ||
                  std::is_same_v<DecayedType, ExternalPtrStmt>) {
      insert(Stmt::make_typed<GlobalStoreStmt>(ptr, data));
    } else {
      TI_ERROR("Statement {} is not a global pointer.", ptr->name());
    }
  }

 private:
  std::unique_ptr<IRNode> root_{nullptr};
  InsertPoint insert_point_;
};

TLANG_NAMESPACE_END
