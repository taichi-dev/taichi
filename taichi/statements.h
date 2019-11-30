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
    TC_ASSERT(inputs.size() == strides.size());
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

  DEFINE_ACCEPT
};

class GetChStmt : public Stmt {
 public:
  Stmt *input_ptr;
  SNode *input_snode, *output_snode;
  int chid;

  GetChStmt(Stmt *input_ptr, int chid);

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
    listgen,
  };

  TaskType task_type;
  SNode *snode;
  int begin, end, step;
  int block_dim;
  bool reversed;
  int num_cpu_threads;
  std::vector<Stmt *> loop_vars;
  std::vector<llvm::Value *> loop_vars_llvm;
  std::unique_ptr<Block> body;

  OffloadedStmt(TaskType task_type) : task_type(task_type) {
    num_cpu_threads = 1;
    begin = end = step = 0;
    block_dim = 0;
    reversed = false;
    if (task_type != TaskType::listgen) {
      body = std::make_unique<Block>();
    }
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

  GlobalTemporaryStmt(std::size_t offset, VectorType ret_type): offset(offset)
    { this->ret_type = ret_type;}

  DEFINE_ACCEPT
};

// Visits all non-containing statements
class BasicStmtVisitor : public IRVisitor {
 public:
  StructForStmt *current_struct_for;

  BasicStmtVisitor() {
    current_struct_for = nullptr;
    allow_undefined_visitor = true;
  }

  void visit(Block *stmt_list) override {
    auto backup_block = current_block;
    current_block = stmt_list;
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    current_block = backup_block;
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
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

  void visit(OffloadedStmt *stmt) override {
    if (stmt->body)
      stmt->body->accept(this);
  }
};

TLANG_NAMESPACE_END
