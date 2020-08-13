#pragma once

#include "taichi/ir/ir.h"
#include "taichi/ir/scratch_pad.h"

TLANG_NAMESPACE_BEGIN

class PragmaSLPStmt : public Stmt {
 public:
  int slp_width;

  PragmaSLPStmt(int slp_width) : slp_width(slp_width) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(slp_width);
  TI_DEFINE_ACCEPT_AND_CLONE
};

class ElementShuffleStmt : public Stmt {
 public:
  LaneAttribute<VectorElement> elements;
  bool pointer;

  ElementShuffleStmt(const LaneAttribute<VectorElement> &elements,
                     bool pointer = false)
      : elements(elements), pointer(pointer) {
    width() = elements.size();
    element_type() = elements[0].stmt->element_type();
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, elements, pointer);
  TI_DEFINE_ACCEPT_AND_CLONE
};

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

class GetRootStmt : public Stmt {
 public:
  GetRootStmt() {
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type);
  TI_DEFINE_ACCEPT_AND_CLONE
};

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

class GetChStmt : public Stmt {
 public:
  Stmt *input_ptr;
  SNode *input_snode, *output_snode;
  int chid;

  GetChStmt(Stmt *input_ptr, int chid);

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, input_ptr, input_snode, output_snode, chid);
  TI_DEFINE_ACCEPT_AND_CLONE
};

class OffloadedStmt : public Stmt {
 public:
  enum TaskType : int {
    serial,
    range_for,
    struct_for,
    clear_list,
    listgen,
    gc,
  };

  TaskType task_type;
  SNode *snode;
  std::size_t begin_offset;
  std::size_t end_offset;
  bool const_begin, const_end;
  int32 begin_value, end_value;
  int step;
  int grid_dim{1};
  int block_dim{1};
  bool reversed;
  int num_cpu_threads;
  Arch device;

  std::vector<int> index_offsets;

  std::unique_ptr<Block> tls_prologue;
  std::unique_ptr<Block> bls_prologue;
  std::unique_ptr<Block> body;
  std::unique_ptr<Block> bls_epilogue;
  std::unique_ptr<Block> tls_epilogue;
  std::size_t tls_size{1};  // avoid allocating dynamic memory with 0 byte
  std::size_t bls_size{0};
  ScratchPadOptions scratch_opt;
  std::unique_ptr<ScratchPads> scratch_pads;

  OffloadedStmt(TaskType task_type);

  OffloadedStmt(TaskType task_type, SNode *snode);

  std::string task_name() const;

  static std::string task_type_name(TaskType tt);

  bool has_body() const {
    return task_type != clear_list && task_type != listgen && task_type != gc;
  }

  bool is_container_statement() const override {
    return has_body();
  }

  std::unique_ptr<Stmt> clone() const override;

  void all_blocks_accept(IRVisitor *visitor);

  TI_STMT_DEF_FIELDS(ret_type,
                     task_type,
                     snode,
                     begin_offset,
                     end_offset,
                     const_begin,
                     const_end,
                     begin_value,
                     end_value,
                     step /*unused?*/,
                     block_dim,
                     reversed,
                     num_cpu_threads,
                     device,
                     index_offsets);
  TI_DEFINE_ACCEPT
};

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

// All loop indices fused together
class LoopLinearIndexStmt : public Stmt {
 public:
  Stmt *loop;
  int index;

  LoopLinearIndexStmt(Stmt *loop) : loop(loop) {
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  // Return the number of bits of the loop, or -1 if unknown.
  // TODO: implement
  // int max_num_bits() const;

  TI_STMT_DEF_FIELDS(ret_type, loop);
  TI_DEFINE_ACCEPT_AND_CLONE
};

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

class GlobalTemporaryStmt : public Stmt {
 public:
  std::size_t offset;

  GlobalTemporaryStmt(std::size_t offset, VectorType ret_type)
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

class ThreadLocalPtrStmt : public Stmt {
 public:
  std::size_t offset;

  ThreadLocalPtrStmt(std::size_t offset, VectorType ret_type) : offset(offset) {
    this->ret_type = ret_type;
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, offset);
  TI_DEFINE_ACCEPT_AND_CLONE
};

class BlockLocalPtrStmt : public Stmt {
 public:
  Stmt *offset;

  BlockLocalPtrStmt(Stmt *offset, VectorType ret_type) : offset(offset) {
    this->ret_type = ret_type;
    TI_STMT_REG_FIELDS;
  }

  bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, offset);
  TI_DEFINE_ACCEPT_AND_CLONE
};

class InternalFuncStmt : public Stmt {
 public:
  std::string func_name;

  InternalFuncStmt(const std::string &func_name) : func_name(func_name) {
    this->ret_type = VectorType(1, DataType::i32);
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(ret_type, func_name);
  TI_DEFINE_ACCEPT_AND_CLONE
};

class StackAllocaStmt : public Stmt {
 public:
  DataType dt;
  std::size_t max_size;  // TODO: 0 = adaptive

  StackAllocaStmt(DataType dt, std::size_t max_size)
      : dt(dt), max_size(max_size) {
    TI_STMT_REG_FIELDS;
  }

  std::size_t element_size_in_bytes() const {
    return data_type_size(ret_type.data_type);
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

class StackLoadTopStmt : public Stmt {
 public:
  Stmt *stack;

  StackLoadTopStmt(Stmt *stack) {
    TI_ASSERT(stack->is<StackAllocaStmt>());
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

class StackLoadTopAdjStmt : public Stmt {
 public:
  Stmt *stack;

  StackLoadTopAdjStmt(Stmt *stack) {
    TI_ASSERT(stack->is<StackAllocaStmt>());
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

class StackPopStmt : public Stmt {
 public:
  Stmt *stack;

  StackPopStmt(Stmt *stack) {
    TI_ASSERT(stack->is<StackAllocaStmt>());
    this->stack = stack;
    TI_STMT_REG_FIELDS;
  }

  // Mark has_global_side_effect == true to prevent being moved out of an if
  // clause in the simplify pass for now.

  TI_STMT_DEF_FIELDS(ret_type, stack);
  TI_DEFINE_ACCEPT_AND_CLONE
};

class StackPushStmt : public Stmt {
 public:
  Stmt *stack;
  Stmt *v;

  StackPushStmt(Stmt *stack, Stmt *v) {
    TI_ASSERT(stack->is<StackAllocaStmt>());
    this->stack = stack;
    this->v = v;
    TI_STMT_REG_FIELDS;
  }

  // Mark has_global_side_effect == true to prevent being moved out of an if
  // clause in the simplify pass for now.

  TI_STMT_DEF_FIELDS(ret_type, stack, v);
  TI_DEFINE_ACCEPT_AND_CLONE
};

class StackAccAdjointStmt : public Stmt {
 public:
  Stmt *stack;
  Stmt *v;

  StackAccAdjointStmt(Stmt *stack, Stmt *v) {
    TI_ASSERT(stack->is<StackAllocaStmt>());
    this->stack = stack;
    this->v = v;
    TI_STMT_REG_FIELDS;
  }

  // Mark has_global_side_effect == true to prevent being moved out of an if
  // clause in the simplify pass for now.

  TI_STMT_DEF_FIELDS(ret_type, stack, v);
  TI_DEFINE_ACCEPT_AND_CLONE
};

TLANG_NAMESPACE_END
