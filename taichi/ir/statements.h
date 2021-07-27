#pragma once

#include "taichi/ir/ir.h"
#include "taichi/ir/offloaded_task_type.h"
#include "taichi/ir/stmt_op_types.h"
#include "taichi/program/arch.h"

namespace taichi {
namespace lang {

class Function;

/**
 * Allocate a local variable with initial value 0.
 */
class AllocaStmt : public Stmt {
 public:
  AllocaStmt(DataType type) {
    ret_type = TypeFactory::create_vector_or_scalar_type(1, type);
    TI_STMT_REG_FIELDS;
  }

  AllocaStmt(int width, DataType type) {
    ret_type = TypeFactory::create_vector_or_scalar_type(width, type);
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  bool common_statement_eliminable() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * Updates mask, break if all bits of the mask are 0.
 */
class WhileControlStmt : public Stmt {
 public:
  Stmt *mask;
  Stmt *cond;
  WhileControlStmt(Stmt *mask, Stmt *cond) : mask(mask), cond(cond) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(mask, cond);
  TI_DEFINE_ACCEPT_AND_CLONE;
};

/**
 * Jump to the next loop iteration, i.e., `continue` in C++.
 */
class ContinueStmt : public Stmt {
 public:
  // This is the loop on which this continue stmt has effects. It can be either
  // an offloaded task, or a for/while loop inside the kernel.
  Stmt *scope;

  ContinueStmt() : scope(nullptr) {
    TI_STMT_REG_FIELDS;
  }

  // For top-level loops, since they are parallelized to multiple threads (on
  // either CPU or GPU), `continue` becomes semantically equivalent to `return`.
  //
  // Caveat:
  // We should wrap each backend's kernel body into a function (as LLVM does).
  // The reason is that, each thread may handle more than one element,
  // depending on the backend's implementation.
  //
  // For example, CUDA uses grid-stride loops, the snippet below illustrates
  // the idea:
  //
  // __global__ foo_kernel(...) {
  //   for (int i = lower; i < upper; i += gridDim) {
  //     auto coord = compute_coords(i);
  //     // run_foo_kernel is produced by codegen
  //     run_foo_kernel(coord);
  //   }
  // }
  //
  // If run_foo_kernel() is directly inlined within foo_kernel(), `return`
  // could prematurely terminate the entire kernel.
  bool as_return() const;

  TI_STMT_DEF_FIELDS(scope);
  TI_DEFINE_ACCEPT_AND_CLONE;
};

/**
 * A unary operation. The field |cast_type| is used only when is_cast() is true.
 */
class UnaryOpStmt : public Stmt {
 public:
  UnaryOpType op_type;
  Stmt *operand;
  DataType cast_type;

  UnaryOpStmt(UnaryOpType op_type, Stmt *operand);

  bool same_operation(UnaryOpStmt *o) const;
  bool is_cast() const;

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, op_type, operand, cast_type);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * Load a kernel argument. The data type should be known when constructing this
 * statement. |is_ptr| should be true iff the result can be used as a base
 * pointer of an ExternalPtrStmt.
 */
class ArgLoadStmt : public Stmt {
 public:
  int arg_id;
  bool is_ptr;

  ArgLoadStmt(int arg_id, const DataType &dt, bool is_ptr = false)
      : arg_id(arg_id) {
    this->ret_type = TypeFactory::create_vector_or_scalar_type(1, dt);
    this->is_ptr = is_ptr;
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, arg_id, is_ptr);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A random value. For i32, u32, i64, and u64, the result is randomly sampled
 * from all possible values with equal probability. For f32 and f64 data types,
 * the result is uniformly sampled in the interval [0, 1).
 * When Taichi runtime initializes, each CUDA thread / CPU thread gets a
 * different (but deterministic as long as the thread id doesn't change)
 * random seed. Each invocation of a RandStmt compiles to a call of a
 * deterministic PRNG to generate a random value in the backend.
 */
class RandStmt : public Stmt {
 public:
  RandStmt(const DataType &dt) {
    ret_type = dt;
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  bool common_statement_eliminable() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A binary operation.
 */
class BinaryOpStmt : public Stmt {
 public:
  BinaryOpType op_type;
  Stmt *lhs, *rhs;
  bool is_bit_vectorized;  // TODO: remove this field

  BinaryOpStmt(BinaryOpType op_type,
               Stmt *lhs,
               Stmt *rhs,
               bool is_bit_vectorized = false)
      : op_type(op_type),
        lhs(lhs),
        rhs(rhs),
        is_bit_vectorized(is_bit_vectorized) {
    TI_ASSERT(!lhs->is<AllocaStmt>());
    TI_ASSERT(!rhs->is<AllocaStmt>());
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, op_type, lhs, rhs, is_bit_vectorized);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A ternary operation. Currently "select" (the ternary conditional operator,
 * "?:" in C++) is the only supported ternary operation.
 */
class TernaryOpStmt : public Stmt {
 public:
  TernaryOpType op_type;
  Stmt *op1, *op2, *op3;

  TernaryOpStmt(TernaryOpType op_type, Stmt *op1, Stmt *op2, Stmt *op3)
      : op_type(op_type), op1(op1), op2(op2), op3(op3) {
    TI_ASSERT(!op1->is<AllocaStmt>());
    TI_ASSERT(!op2->is<AllocaStmt>());
    TI_ASSERT(!op3->is<AllocaStmt>());
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, op1, op2, op3);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * An atomic operation.
 */
class AtomicOpStmt : public Stmt {
 public:
  AtomicOpType op_type;
  Stmt *dest, *val;
  bool is_reduction;

  AtomicOpStmt(AtomicOpType op_type, Stmt *dest, Stmt *val)
      : op_type(op_type), dest(dest), val(val), is_reduction(false) {
    TI_STMT_REG_FIELDS;
  }

  static std::unique_ptr<AtomicOpStmt> make_for_reduction(AtomicOpType op_type,
                                                          Stmt *dest,
                                                          Stmt *val) {
    auto stmt = std::make_unique<AtomicOpStmt>(op_type, dest, val);
    stmt->is_reduction = true;
    return stmt;
  }

  TI_STMT_DEF_FIELDS(ret_type, op_type, dest, val);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * An external pointer. |base_ptrs| should be ArgLoadStmts with
 * |is_ptr| == true.
 */
class ExternalPtrStmt : public Stmt {
 public:
  LaneAttribute<Stmt *> base_ptrs;
  std::vector<Stmt *> indices;

  ExternalPtrStmt(const LaneAttribute<Stmt *> &base_ptrs,
                  const std::vector<Stmt *> &indices);

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, base_ptrs, indices);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A global pointer, currently only able to represent an address in a SNode.
 * When |activate| is true, this statement activates the address it points to,
 * so it has "global side effect" in this case.
 * After the "lower_access" pass, all GlobalPtrStmts should be lowered into
 * SNodeLookupStmts and GetChStmts, and should not appear in the final lowered
 * IR.
 */
class GlobalPtrStmt : public Stmt {
 public:
  LaneAttribute<SNode *> snodes;
  std::vector<Stmt *> indices;
  bool activate;
  bool is_bit_vectorized;  // for bit_loop_vectorize pass

  GlobalPtrStmt(const LaneAttribute<SNode *> &snodes,
                const std::vector<Stmt *> &indices,
                bool activate = true);

  bool is_element_wise(const SNode *snode) const;

  bool covers_snode(const SNode *snode) const;

  bool has_global_side_effect() const override {
    return activate;
  }

  bool common_statement_eliminable() const override {
    return true;
  }

  TI_STMT_DEF_FIELDS(ret_type, snodes, indices, activate, is_bit_vectorized);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * An operation to a SNode (not necessarily a leaf SNode).
 */
class SNodeOpStmt : public Stmt {
 public:
  SNodeOpType op_type;
  SNode *snode;
  Stmt *ptr;
  Stmt *val;

  SNodeOpStmt(SNodeOpType op_type,
              SNode *snode,
              Stmt *ptr,
              Stmt *val = nullptr);

  static bool activation_related(SNodeOpType op);

  static bool need_activation(SNodeOpType op);

  TI_STMT_DEF_FIELDS(ret_type, op_type, snode, ptr, val);
  TI_DEFINE_ACCEPT_AND_CLONE
};

// TODO: remove this
class ExternalTensorShapeAlongAxisStmt : public Stmt {
 public:
  int axis;
  int arg_id;

  ExternalTensorShapeAlongAxisStmt(int axis, int arg_id);

  TI_STMT_DEF_FIELDS(ret_type, axis, arg_id);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * An assertion.
 * If |cond| is false, print the formatted |text| with |args|, and terminate
 * the program.
 */
class AssertStmt : public Stmt {
 public:
  Stmt *cond;
  std::string text;
  std::vector<Stmt *> args;

  AssertStmt(Stmt *cond,
             const std::string &text,
             const std::vector<Stmt *> &args)
      : cond(cond), text(text), args(args) {
    TI_ASSERT(cond);
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(cond, text, args);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * Call an external (C++) function.
 */
class ExternalFuncCallStmt : public Stmt {
 public:
  void *func;
  std::string source;
  std::vector<Stmt *> arg_stmts;
  std::vector<Stmt *> output_stmts;

  ExternalFuncCallStmt(void *func,
                       const std::string &source,
                       const std::vector<Stmt *> &arg_stmts,
                       const std::vector<Stmt *> &output_stmts)
      : func(func),
        source(source),
        arg_stmts(arg_stmts),
        output_stmts(output_stmts) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(func, arg_stmts, output_stmts);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A hint to the Taichi compiler about the relation of the values of two
 * statements.
 * This statement simply returns the input statement at the backend, and hints
 * the Taichi compiler that |base| + |low| <= |input| < |base| + |high|.
 */
class RangeAssumptionStmt : public Stmt {
 public:
  Stmt *input;
  Stmt *base;
  int low, high;

  RangeAssumptionStmt(Stmt *input, Stmt *base, int low, int high)
      : input(input), base(base), low(low), high(high) {
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, input, base, low, high);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A hint to the Taichi compiler that a statement has unique values among
 * the top-level loop. This statement simply returns the input statement at
 * the backend, and hints the Taichi compiler that this statement never
 * evaluate to the same value across different iterations of the top-level
 * loop. This statement's value set among all iterations of the top-level loop
 * also covers all active indices of each SNodes with id in the |covers| field
 * of this statement. Since this statement can only evaluate to one value,
 * the SNodes with id in the |covers| field should have only one dimension.
 */
class LoopUniqueStmt : public Stmt {
 public:
  Stmt *input;
  std::unordered_set<int> covers;  // Stores SNode id
  // std::unordered_set<> provides operator==, and StmtFieldManager will
  // use that to check if two LoopUniqueStmts are the same.

  LoopUniqueStmt(Stmt *input, const std::vector<SNode *> &covers);

  bool covers_snode(const SNode *snode) const;

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, input, covers);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A load from a global address, including SNodes, external arrays, TLS, BLS,
 * and global temporary variables.
 */
class GlobalLoadStmt : public Stmt {
 public:
  Stmt *src;

  explicit GlobalLoadStmt(Stmt *src) : src(src) {
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  bool common_statement_eliminable() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, src);
  TI_DEFINE_ACCEPT_AND_CLONE;
};

/**
 * A store to a global address, including SNodes, external arrays, TLS, BLS,
 * and global temporary variables.
 */
class GlobalStoreStmt : public Stmt {
 public:
  Stmt *dest;
  Stmt *val;

  GlobalStoreStmt(Stmt *dest, Stmt *val) : dest(dest), val(val) {
    TI_STMT_REG_FIELDS;
  }

  bool common_statement_eliminable() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, dest, val);
  TI_DEFINE_ACCEPT_AND_CLONE;
};

/**
 * A load from a local variable, i.e., an "alloca".
 */
class LocalLoadStmt : public Stmt {
 public:
  LaneAttribute<LocalAddress> src;

  explicit LocalLoadStmt(const LaneAttribute<LocalAddress> &src) : src(src) {
    TI_STMT_REG_FIELDS;
  }

  bool same_source() const;
  bool has_source(Stmt *alloca) const;

  Stmt *previous_store_or_alloca_in_block();

  bool has_global_side_effect() const override {
    return false;
  }

  bool common_statement_eliminable() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, src);
  TI_DEFINE_ACCEPT_AND_CLONE;
};

/**
 * A store to a local variable, i.e., an "alloca".
 */
class LocalStoreStmt : public Stmt {
 public:
  Stmt *dest;
  Stmt *val;

  LocalStoreStmt(Stmt *dest, Stmt *val) : dest(dest), val(val) {
    TI_ASSERT(dest->is<AllocaStmt>());
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  bool dead_instruction_eliminable() const override {
    return false;
  }

  bool common_statement_eliminable() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, dest, val);
  TI_DEFINE_ACCEPT_AND_CLONE;
};

/**
 * Same as "if (cond) true_statements; else false_statements;" in C++.
 * |true_mask| and |false_mask| are used to support vectorization.
 */
class IfStmt : public Stmt {
 public:
  Stmt *cond;
  Stmt *true_mask, *false_mask;
  std::unique_ptr<Block> true_statements, false_statements;

  explicit IfStmt(Stmt *cond);

  // Use these setters to set Block::parent_stmt at the same time.
  void set_true_statements(std::unique_ptr<Block> &&new_true_statements);
  void set_false_statements(std::unique_ptr<Block> &&new_false_statements);

  bool is_container_statement() const override {
    return true;
  }

  std::unique_ptr<Stmt> clone() const override;

  TI_STMT_DEF_FIELDS(cond, true_mask, false_mask);
  TI_DEFINE_ACCEPT
};

/**
 * Print the contents in this statement. Each entry in the contents can be
 * either a statement or a string, and they are printed one by one, separated
 * by a comma and a space.
 */
class PrintStmt : public Stmt {
 public:
  using EntryType = std::variant<Stmt *, std::string>;
  std::vector<EntryType> contents;

  PrintStmt(const std::vector<EntryType> &contents_) : contents(contents_) {
    TI_STMT_REG_FIELDS;
  }

  template <typename... Args>
  PrintStmt(Stmt *t, Args &&... args)
      : contents(make_entries(t, std::forward<Args>(args)...)) {
    TI_STMT_REG_FIELDS;
  }

  template <typename... Args>
  PrintStmt(const std::string &str, Args &&... args)
      : contents(make_entries(str, std::forward<Args>(args)...)) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(ret_type, contents);
  TI_DEFINE_ACCEPT_AND_CLONE

 private:
  static void make_entries_helper(std::vector<PrintStmt::EntryType> &entries) {
  }

  template <typename T, typename... Args>
  static void make_entries_helper(std::vector<PrintStmt::EntryType> &entries,
                                  T &&t,
                                  Args &&... values) {
    entries.push_back(EntryType{t});
    make_entries_helper(entries, std::forward<Args>(values)...);
  }

  template <typename... Args>
  static std::vector<EntryType> make_entries(Args &&... values) {
    std::vector<EntryType> ret;
    make_entries_helper(ret, std::forward<Args>(values)...);
    return ret;
  }
};

/**
 * A constant value.
 */
class ConstStmt : public Stmt {
 public:
  LaneAttribute<TypedConstant> val;

  explicit ConstStmt(const LaneAttribute<TypedConstant> &val) : val(val) {
    TI_ASSERT(val.size() == 1);  // TODO: support vectorized case
    ret_type = val[0].dt;
    for (int i = 0; i < val.size(); i++) {
      TI_ASSERT(val[0].dt == val[i].dt);
    }
    TI_STMT_REG_FIELDS;
  }

  void repeat(int factor) override {
    Stmt::repeat(factor);
    val.repeat(factor);
  }

  bool has_global_side_effect() const override {
    return false;
  }

  std::unique_ptr<ConstStmt> copy();

  TI_STMT_DEF_FIELDS(ret_type, val);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A general range for, similar to "for (i = begin; i < end; i++) body;" in C++.
 * When |reversed| is true, the for loop is reversed, i.e.,
 * "for (i = end - 1; i >= begin; i--) body;".
 * When the statement is in the top level before offloading, it will be
 * offloaded to a parallel for loop. Otherwise, it will be offloaded to a
 * serial for loop.
 */
class RangeForStmt : public Stmt {
 public:
  Stmt *begin, *end;
  std::unique_ptr<Block> body;
  bool reversed;
  int vectorize;
  int bit_vectorize;
  int num_cpu_threads;
  int block_dim;
  bool strictly_serialized;

  RangeForStmt(Stmt *begin,
               Stmt *end,
               std::unique_ptr<Block> &&body,
               int vectorize,
               int bit_vectorize,
               int num_cpu_threads,
               int block_dim,
               bool strictly_serialized);

  bool is_container_statement() const override {
    return true;
  }

  void reverse() {
    reversed = !reversed;
  }

  std::unique_ptr<Stmt> clone() const override;

  TI_STMT_DEF_FIELDS(begin,
                     end,
                     reversed,
                     vectorize,
                     bit_vectorize,
                     num_cpu_threads,
                     block_dim,
                     strictly_serialized);
  TI_DEFINE_ACCEPT
};

/**
 * A parallel for loop over a SNode, similar to "for i in snode: body"
 * in Python. This statement must be at the top level before offloading.
 */
class StructForStmt : public Stmt {
 public:
  SNode *snode;
  std::unique_ptr<Block> body;
  std::unique_ptr<Block> block_initialization;
  std::unique_ptr<Block> block_finalization;
  std::vector<int> index_offsets;
  int vectorize;
  int bit_vectorize;
  int num_cpu_threads;
  int block_dim;
  MemoryAccessOptions mem_access_opt;

  StructForStmt(SNode *snode,
                std::unique_ptr<Block> &&body,
                int vectorize,
                int bit_vectorize,
                int num_cpu_threads,
                int block_dim);

  bool is_container_statement() const override {
    return true;
  }

  std::unique_ptr<Stmt> clone() const override;

  TI_STMT_DEF_FIELDS(snode,
                     index_offsets,
                     vectorize,
                     bit_vectorize,
                     num_cpu_threads,
                     block_dim,
                     mem_access_opt);
  TI_DEFINE_ACCEPT
};

/**
 * An inline Taichi function.
 * TODO: This statement seems unused.
 */
class FuncBodyStmt : public Stmt {
 public:
  std::string funcid;
  std::unique_ptr<Block> body;

  FuncBodyStmt(const std::string &funcid, std::unique_ptr<Block> &&body);

  bool is_container_statement() const override {
    return true;
  }

  std::unique_ptr<Stmt> clone() const override;

  TI_STMT_DEF_FIELDS(funcid);
  TI_DEFINE_ACCEPT
};

/**
 * Call an inline Taichi function.
 */
class FuncCallStmt : public Stmt {
 public:
  Function *func;
  std::vector<Stmt *> args;

  FuncCallStmt(Function *func, const std::vector<Stmt *> &args);

  TI_STMT_DEF_FIELDS(ret_type, func, args);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * Exit the kernel or function with a return value.
 */
class ReturnStmt : public Stmt {
 public:
  Stmt *value;

  explicit ReturnStmt(Stmt *value) : value(value) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(value);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A serial while-true loop. |mask| is to support vectorization.
 */
class WhileStmt : public Stmt {
 public:
  Stmt *mask;
  std::unique_ptr<Block> body;

  explicit WhileStmt(std::unique_ptr<Block> &&body);

  bool is_container_statement() const override {
    return true;
  }

  std::unique_ptr<Stmt> clone() const override;

  TI_STMT_DEF_FIELDS(mask);
  TI_DEFINE_ACCEPT
};

// TODO: remove this
class PragmaSLPStmt : public Stmt {
 public:
  int slp_width;

  PragmaSLPStmt(int slp_width) : slp_width(slp_width) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(slp_width);
  TI_DEFINE_ACCEPT_AND_CLONE
};

// TODO: document for this
class ElementShuffleStmt : public Stmt {
 public:
  LaneAttribute<VectorElement> elements;
  bool pointer;

  explicit ElementShuffleStmt(const LaneAttribute<VectorElement> &elements,
                              bool pointer = false)
      : elements(elements), pointer(pointer) {
    TI_ASSERT(elements.size() == 1);  // TODO: support vectorized cases
    ret_type = elements[0].stmt->element_type();
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, elements, pointer);
  TI_DEFINE_ACCEPT_AND_CLONE
};

// TODO: remove this (replace with input + ConstStmt(offset))
class IntegerOffsetStmt : public Stmt {
 public:
  Stmt *input;
  int64 offset;

  IntegerOffsetStmt(Stmt *input, int64 offset) : input(input), offset(offset) {
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, input, offset);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * All indices of an address fused together.
 */
class LinearizeStmt : public Stmt {
 public:
  std::vector<Stmt *> inputs;
  std::vector<int> strides;

  LinearizeStmt(const std::vector<Stmt *> &inputs,
                const std::vector<int> &strides)
      : inputs(inputs), strides(strides) {
    TI_ASSERT(inputs.size() == strides.size());
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, inputs, strides);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * Extract an interval of bits from an integral value.
 * Equivalent to (|input| >> |bit_begin|) &
 *   ((1 << (|bit_end| - |bit_begin|)) - 1).
 */
class BitExtractStmt : public Stmt {
 public:
  Stmt *input;
  int bit_begin, bit_end;
  bool simplified;
  BitExtractStmt(Stmt *input, int bit_begin, int bit_end)
      : input(input), bit_begin(bit_begin), bit_end(bit_end) {
    simplified = false;
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, input, bit_begin, bit_end, simplified);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * The SNode root.
 */
class GetRootStmt : public Stmt {
 public:
  GetRootStmt(SNode *root = nullptr) : root_(root) {
    if (this->root_ != nullptr) {
      while (this->root_->parent) {
        this->root_ = this->root_->parent;
      }
    }
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, root_);
  TI_DEFINE_ACCEPT_AND_CLONE

  SNode *root() {
    return root_;
  }

 private:
  SNode *root_;
};

/**
 * Lookup a component of a SNode.
 */
class SNodeLookupStmt : public Stmt {
 public:
  SNode *snode;
  Stmt *input_snode;
  Stmt *input_index;
  bool activate;

  SNodeLookupStmt(SNode *snode,
                  Stmt *input_snode,
                  Stmt *input_index,
                  bool activate)
      : snode(snode),
        input_snode(input_snode),
        input_index(input_index),
        activate(activate) {
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return activate;
  }

  bool common_statement_eliminable() const override {
    return true;
  }

  TI_STMT_DEF_FIELDS(ret_type, snode, input_snode, input_index, activate);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * Get a child of a SNode on the hierarchical SNode tree.
 */
class GetChStmt : public Stmt {
 public:
  Stmt *input_ptr;
  SNode *input_snode, *output_snode;
  int chid;
  bool is_bit_vectorized;

  GetChStmt(Stmt *input_ptr, int chid, bool is_bit_vectorized = false);

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type,
                     input_ptr,
                     input_snode,
                     output_snode,
                     chid,
                     is_bit_vectorized);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * The statement corresponding to an offloaded task.
 */
class OffloadedStmt : public Stmt {
 public:
  using TaskType = OffloadedTaskType;

  TaskType task_type;
  Arch device;
  SNode *snode{nullptr};
  std::size_t begin_offset{0};
  std::size_t end_offset{0};
  bool const_begin{false};
  bool const_end{false};
  int32 begin_value{0};
  int32 end_value{0};
  int grid_dim{1};
  int block_dim{1};
  bool reversed{false};
  int num_cpu_threads{1};

  std::vector<int> index_offsets;

  std::unique_ptr<Block> tls_prologue;
  std::unique_ptr<Block> bls_prologue;
  std::unique_ptr<Block> body;
  std::unique_ptr<Block> bls_epilogue;
  std::unique_ptr<Block> tls_epilogue;
  std::size_t tls_size{1};  // avoid allocating dynamic memory with 0 byte
  std::size_t bls_size{0};
  MemoryAccessOptions mem_access_opt;

  OffloadedStmt(TaskType task_type, Arch arch);

  std::string task_name() const;

  static std::string task_type_name(TaskType tt);

  bool has_body() const {
    return task_type != TaskType::listgen && task_type != TaskType::gc;
  }

  bool is_container_statement() const override {
    return has_body();
  }

  std::unique_ptr<Stmt> clone() const override;

  void all_blocks_accept(IRVisitor *visitor);

  TI_STMT_DEF_FIELDS(ret_type /*inherited from Stmt*/,
                     task_type,
                     device,
                     snode,
                     begin_offset,
                     end_offset,
                     const_begin,
                     const_end,
                     begin_value,
                     end_value,
                     grid_dim,
                     block_dim,
                     reversed,
                     num_cpu_threads,
                     index_offsets,
                     mem_access_opt);
  TI_DEFINE_ACCEPT
};

/**
 * The |index|-th index of the |loop|.
 */
class LoopIndexStmt : public Stmt {
 public:
  Stmt *loop;
  int index;

  LoopIndexStmt(Stmt *loop, int index) : loop(loop), index(index) {
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  // Return the number of bits of the loop, or -1 if unknown.
  int max_num_bits() const;

  TI_STMT_DEF_FIELDS(ret_type, loop, index);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * thread index within a CUDA block
 * TODO: Remove this. Have a better way for retrieving thread index.
 */
class LoopLinearIndexStmt : public Stmt {
 public:
  Stmt *loop;

  explicit LoopLinearIndexStmt(Stmt *loop) : loop(loop) {
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, loop);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * The lowest |index|-th index of the |loop| among the iterations iterated by
 * the block.
 */
class BlockCornerIndexStmt : public Stmt {
 public:
  Stmt *loop;
  int index;

  BlockCornerIndexStmt(Stmt *loop, int index) : loop(loop), index(index) {
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, loop, index);
  TI_DEFINE_ACCEPT_AND_CLONE
};

// TODO: remove this
class BlockDimStmt : public Stmt {
 public:
  BlockDimStmt() {
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A global temporary variable, located at |offset| in the global temporary
 * buffer.
 */
class GlobalTemporaryStmt : public Stmt {
 public:
  std::size_t offset;

  GlobalTemporaryStmt(std::size_t offset, const DataType &ret_type)
      : offset(offset) {
    this->ret_type = ret_type;
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, offset);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A thread-local pointer, located at |offset| in the thread-local storage.
 */
class ThreadLocalPtrStmt : public Stmt {
 public:
  std::size_t offset;

  ThreadLocalPtrStmt(std::size_t offset, const DataType &ret_type)
      : offset(offset) {
    this->ret_type = ret_type;
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, offset);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A block-local pointer, located at |offset| in the block-local storage.
 */
class BlockLocalPtrStmt : public Stmt {
 public:
  Stmt *offset;

  BlockLocalPtrStmt(Stmt *offset, const DataType &ret_type) : offset(offset) {
    this->ret_type = ret_type;
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, offset);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * The statement corresponding to a clear-list task.
 */
class ClearListStmt : public Stmt {
 public:
  explicit ClearListStmt(SNode *snode);

  SNode *snode;

  TI_STMT_DEF_FIELDS(ret_type, snode);
  TI_DEFINE_ACCEPT_AND_CLONE
};

// Checks if the task represented by |stmt| contains a single ClearListStmt.
bool is_clear_list_task(const OffloadedStmt *stmt);

// TODO: remove this
class InternalFuncStmt : public Stmt {
 public:
  std::string func_name;

  explicit InternalFuncStmt(const std::string &func_name)
      : func_name(func_name) {
    this->ret_type =
        TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32);
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(ret_type, func_name);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A local AD-stack.
 */
class AdStackAllocaStmt : public Stmt {
 public:
  DataType dt;
  std::size_t max_size{0};  // 0 = adaptive

  AdStackAllocaStmt(const DataType &dt, std::size_t max_size)
      : dt(dt), max_size(max_size) {
    TI_STMT_REG_FIELDS;
  }

  std::size_t element_size_in_bytes() const {
    return data_type_size(ret_type);
  }

  std::size_t entry_size_in_bytes() const {
    return element_size_in_bytes() * 2;
  }

  std::size_t size_in_bytes() const {
    return sizeof(int32) + entry_size_in_bytes() * max_size;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  bool common_statement_eliminable() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, dt, max_size);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * Load the top primal value of an AD-stack.
 */
class AdStackLoadTopStmt : public Stmt {
 public:
  Stmt *stack;

  explicit AdStackLoadTopStmt(Stmt *stack) {
    TI_ASSERT(stack->is<AdStackAllocaStmt>());
    this->stack = stack;
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  bool common_statement_eliminable() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, stack);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * Load the top adjoint value of an AD-stack.
 */
class AdStackLoadTopAdjStmt : public Stmt {
 public:
  Stmt *stack;

  explicit AdStackLoadTopAdjStmt(Stmt *stack) {
    TI_ASSERT(stack->is<AdStackAllocaStmt>());
    this->stack = stack;
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  bool common_statement_eliminable() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, stack);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * Pop the top primal and adjoint values in the AD-stack.
 */
class AdStackPopStmt : public Stmt {
 public:
  Stmt *stack;

  explicit AdStackPopStmt(Stmt *stack) {
    TI_ASSERT(stack->is<AdStackAllocaStmt>());
    this->stack = stack;
    TI_STMT_REG_FIELDS;
  }

  // Mark has_global_side_effect == true to prevent being moved out of an if
  // clause in the simplify pass for now.

  TI_STMT_DEF_FIELDS(ret_type, stack);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * Push a primal value to the AD-stack, and set the corresponding adjoint
 * value to 0.
 */
class AdStackPushStmt : public Stmt {
 public:
  Stmt *stack;
  Stmt *v;

  AdStackPushStmt(Stmt *stack, Stmt *v) {
    TI_ASSERT(stack->is<AdStackAllocaStmt>());
    this->stack = stack;
    this->v = v;
    TI_STMT_REG_FIELDS;
  }

  // Mark has_global_side_effect == true to prevent being moved out of an if
  // clause in the simplify pass for now.

  TI_STMT_DEF_FIELDS(ret_type, stack, v);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * Accumulate |v| to the top adjoint value of the AD-stack.
 */
class AdStackAccAdjointStmt : public Stmt {
 public:
  Stmt *stack;
  Stmt *v;

  AdStackAccAdjointStmt(Stmt *stack, Stmt *v) {
    TI_ASSERT(stack->is<AdStackAllocaStmt>());
    this->stack = stack;
    this->v = v;
    TI_STMT_REG_FIELDS;
  }

  // Mark has_global_side_effect == true to prevent being moved out of an if
  // clause in the simplify pass for now.

  TI_STMT_DEF_FIELDS(ret_type, stack, v);
  TI_DEFINE_ACCEPT_AND_CLONE
};

/**
 * A global store to one or more children of a bit struct.
 */
class BitStructStoreStmt : public Stmt {
 public:
  Stmt *ptr;
  std::vector<int> ch_ids;
  std::vector<Stmt *> values;
  bool is_atomic;

  BitStructStoreStmt(Stmt *ptr,
                     const std::vector<int> &ch_ids,
                     const std::vector<Stmt *> &values)
      : ptr(ptr), ch_ids(ch_ids), values(values), is_atomic(true) {
    TI_ASSERT(ch_ids.size() == values.size());
    TI_STMT_REG_FIELDS;
  }

  SNode *get_bit_struct_snode() const;

  bool common_statement_eliminable() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, ptr, ch_ids, values, is_atomic);
  TI_DEFINE_ACCEPT_AND_CLONE;
};

}  // namespace lang
}  // namespace taichi
