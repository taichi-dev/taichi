#pragma once
#include "ir.h"

TLANG_NAMESPACE_BEGIN

class PragmaSLPStmt : public Stmt {
 public:
  int slp_width;

  PragmaSLPStmt(int slp_width) : slp_width(slp_width) {
  }

  DEFINE_ACCEPT
};

class VectorElement {
 public:
  Stmt *stmt;
  int index;

  VectorElement() : stmt(nullptr), index(0) {
  }

  VectorElement(Stmt *stmt, int index) : stmt(stmt), index(index) {
  }
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
    for (int i = 0; i < width(); i++) {
      add_operand(this->elements[i].stmt);
    }
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }
  DEFINE_ACCEPT
};

class IntegerOffsetStmt : public Stmt {
 public:
  Stmt *input;
  int64 offset;

  IntegerOffsetStmt(Stmt *input, int64 offset) : input(input), offset(offset) {
    add_operand(this->input);
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }
  DEFINE_ACCEPT
};

class LinearizeStmt : public Stmt {
 public:
  std::vector<Stmt *> inputs;
  std::vector<int> strides;

  LinearizeStmt(const std::vector<Stmt *> &inputs,
                const std::vector<int> &strides)
      : inputs(inputs), strides(strides) {
    TI_ASSERT(inputs.size() == strides.size());
    for (auto &op : this->inputs) {
      add_operand(op);
    }
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }
  DEFINE_ACCEPT
};

class OffsetAndExtractBitsStmt : public Stmt {
 public:
  Stmt *input;
  int bit_begin, bit_end;
  int64 offset;
  bool simplified;
  OffsetAndExtractBitsStmt(Stmt *input, int bit_begin, int bit_end, int offset)
      : input(input), bit_begin(bit_begin), bit_end(bit_end), offset(offset) {
    add_operand(this->input);
    simplified = false;
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }
  DEFINE_ACCEPT;
};

class GetRootStmt : public Stmt {
 public:
  GetRootStmt() {
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }
  DEFINE_ACCEPT
};

class SNodeLookupStmt : public Stmt {
 public:
  SNode *snode;
  Stmt *input_snode;
  Stmt *input_index;
  std::vector<Stmt *> global_indices;
  bool activate;

  SNodeLookupStmt(SNode *snode,
                  Stmt *input_snode,
                  Stmt *input_index,
                  bool activate,
                  const std::vector<Stmt *> &global_indices)
      : snode(snode),
        input_snode(input_snode),
        input_index(input_index),
        global_indices(global_indices),
        activate(activate) {
    add_operand(this->input_snode);
    add_operand(this->input_index);
    for (int i = 0; i < (int)global_indices.size(); i++) {
      add_operand(this->global_indices[i]);
    }
  }

  virtual bool has_global_side_effect() const override {
    return activate;
  }

  DEFINE_ACCEPT
};

class GetChStmt : public Stmt {
 public:
  Stmt *input_ptr;
  SNode *input_snode, *output_snode;
  int chid;

  GetChStmt(Stmt *input_ptr, int chid);

  virtual bool has_global_side_effect() const override {
    return false;
  }

  DEFINE_ACCEPT
};

class ClearAllStmt : public Stmt {
 public:
  SNode *snode;
  bool deactivate;

  ClearAllStmt(SNode *snode, bool deactivate)
      : snode(snode), deactivate(deactivate) {
  }

  DEFINE_ACCEPT
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
  Stmt *begin_stmt, *end_stmt;
  std::size_t begin_offset;
  std::size_t end_offset;
  bool const_begin, const_end;
  int32 begin_value, end_value;
  int step;
  int block_dim;
  bool reversed;
  int num_cpu_threads;
  Arch device;
  std::vector<Stmt *> loop_vars;
  std::vector<llvm::Value *> loop_vars_llvm;
  std::unique_ptr<Block> body;

  OffloadedStmt(TaskType task_type);

  OffloadedStmt(TaskType task_type, SNode *snode);

  std::string task_name() const;

  bool has_body() const {
    return task_type != clear_list && task_type != listgen && task_type != gc;
  }

  DEFINE_ACCEPT
};

class LoopIndexStmt : public Stmt {
 public:
  int index;
  bool is_struct_for;

  LoopIndexStmt(int index, bool is_struct_for)
      : index(index), is_struct_for(is_struct_for) {
  }

  DEFINE_ACCEPT
};

class GlobalTemporaryStmt : public Stmt {
 public:
  std::size_t offset;

  GlobalTemporaryStmt(std::size_t offset, VectorType ret_type)
      : offset(offset) {
    this->ret_type = ret_type;
  }

  DEFINE_ACCEPT
};

class InternalFuncStmt : public Stmt {
 public:
  std::string func_name;

  InternalFuncStmt(const std::string &func_name) : func_name(func_name) {
    this->ret_type = VectorType(1, DataType::i32);
  }

  DEFINE_ACCEPT
};

class StackAllocaStmt : public Stmt {
 public:
  DataType dt;
  std::size_t max_size;  // TODO: 0 = adaptive

  StackAllocaStmt(DataType dt, std::size_t max_size)
      : dt(dt), max_size(max_size) {
  }

  DEFINE_ACCEPT
};

class StackLoadTopStmt : public Stmt {
 public:
  Stmt *stack;

  StackLoadTopStmt(Stmt *stack) {
    TI_ASSERT(stack->is<StackAllocaStmt>());
    this->stack = stack;
    add_operand(this->stack);
  }

  DEFINE_ACCEPT
};

class StackLoadTopAdjStmt : public Stmt {
 public:
  Stmt *stack;

  StackLoadTopAdjStmt(Stmt *stack) {
    TI_ASSERT(stack->is<StackAllocaStmt>());
    this->stack = stack;
    add_operand(this->stack);
  }

  DEFINE_ACCEPT
};

class StackPopStmt : public Stmt {
 public:
  Stmt *stack;

  StackPopStmt(Stmt *stack) {
    TI_ASSERT(stack->is<StackAllocaStmt>());
    this->stack = stack;
    add_operand(this->stack);
  }

  DEFINE_ACCEPT
};

class StackPushStmt : public Stmt {
 public:
  Stmt *stack;
  Stmt *v;

  StackPushStmt(Stmt *stack, Stmt *v) {
    TI_ASSERT(stack->is<StackAllocaStmt>());
    this->stack = stack;
    this->v = v;
    add_operand(this->stack);
    add_operand(this->v);
  }

  DEFINE_ACCEPT
};

class StackAccAdjointStmt : public Stmt {
 public:
  Stmt *stack;
  Stmt *v;

  StackAccAdjointStmt(Stmt *stack, Stmt *v) {
    TI_ASSERT(stack->is<StackAllocaStmt>());
    this->stack = stack;
    this->v = v;
    add_operand(this->stack);
    add_operand(this->v);
  }
};

TLANG_NAMESPACE_END
