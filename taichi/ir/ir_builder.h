#pragma once

#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

class IRBuilder {
 private:
  struct InsertPoint {
    Block *block;
    int position;
  };

  std::unique_ptr<IRNode> root_;
  InsertPoint insert_point_;

 public:
  IRBuilder();

  // General inserter. Returns stmt.get().
  Stmt *insert(std::unique_ptr<Stmt> &&stmt);

  // Constants. TODO: add more types
  Stmt *get_int32(int32 value);
  Stmt *get_int64(int64 value);
  Stmt *get_float32(float32 value);
  Stmt *get_float64(float64 value);

  // Load kernel arguments.
  Stmt *create_arg_load(int arg_id, DataType dt, bool is_ptr);

  // The return value of the kernel.
  Stmt *create_return(Stmt *value);

  // Unary operations. Returns the result.
  Stmt *create_cast(Stmt *value, DataType output_type);  // cast by value
  Stmt *create_bit_cast(Stmt *value, DataType output_type);
  Stmt *create_neg(Stmt *value);
  Stmt *create_not(Stmt *value);  // bitwise
  Stmt *create_logical_not(Stmt *value);
  Stmt *create_floor(Stmt *value);
  Stmt *create_ceil(Stmt *value);
  Stmt *create_abs(Stmt *value);
  Stmt *create_sgn(Stmt *value);
  Stmt *create_sqrt(Stmt *value);
  Stmt *create_rsqrt(Stmt *value);
  Stmt *create_sin(Stmt *value);
  Stmt *create_asin(Stmt *value);
  Stmt *create_cos(Stmt *value);
  Stmt *create_acos(Stmt *value);
  Stmt *create_tan(Stmt *value);
  Stmt *create_tanh(Stmt *value);
  Stmt *create_exp(Stmt *value);
  Stmt *create_log(Stmt *value);

  // Binary operations. Returns the result.
  Stmt *create_add(Stmt *l, Stmt *r);
  Stmt *create_sub(Stmt *l, Stmt *r);
  Stmt *create_mul(Stmt *l, Stmt *r);
  // l / r in C++
  Stmt *create_div(Stmt *l, Stmt *r);
  // floor(1.0 * l / r) in C++
  Stmt *create_floordiv(Stmt *l, Stmt *r);
  // 1.0 * l / r in C++
  Stmt *create_truediv(Stmt *l, Stmt *r);
  Stmt *create_mod(Stmt *l, Stmt *r);
  Stmt *create_max(Stmt *l, Stmt *r);
  Stmt *create_min(Stmt *l, Stmt *r);
  Stmt *create_atan2(Stmt *l, Stmt *r);
  Stmt *create_pow(Stmt *l, Stmt *r);
  // Bitwise operations. TODO: add logical operations when we support them
  Stmt *create_and(Stmt *l, Stmt *r);
  Stmt *create_or(Stmt *l, Stmt *r);
  Stmt *create_xor(Stmt *l, Stmt *r);
  Stmt *create_shl(Stmt *l, Stmt *r);
  Stmt *create_shr(Stmt *l, Stmt *r);
  Stmt *create_sar(Stmt *l, Stmt *r);
  // Comparisons.
  Stmt *create_cmp_lt(Stmt *l, Stmt *r);
  Stmt *create_cmp_le(Stmt *l, Stmt *r);
  Stmt *create_cmp_gt(Stmt *l, Stmt *r);
  Stmt *create_cmp_ge(Stmt *l, Stmt *r);
  Stmt *create_cmp_eq(Stmt *l, Stmt *r);
  Stmt *create_cmp_ne(Stmt *l, Stmt *r);

  // Ternary operations. Returns the result.
  Stmt *create_select(Stmt *cond, Stmt *true_result, Stmt *false_result);

  // Print values and strings. Arguments can be Stmt* or std::string.
  template <typename... Args>
  Stmt *create_print(Args &&... args) {
    return insert(Stmt::make<PrintStmt>(std::forward(args)...));
  }


};

TLANG_NAMESPACE_END
