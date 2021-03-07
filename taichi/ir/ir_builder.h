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

  // Print values and strings. Arguments can be Stmt* or std::string.
  template <typename... Args>
  Stmt *create_print(Args &&... args) {
    return insert(Stmt::make<PrintStmt>(std::forward(args)));
  }
};

TLANG_NAMESPACE_END
