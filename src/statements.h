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
  enum Type : int {
    serial,
    range_for,
    struct_for,
    listgen,
  };

  Type type;
  SNode *snode;
  std::unique_ptr<Block> body_block;
  std::unique_ptr<Stmt> body_stmt;

  OffloadedStmt(Type type) : type(type) {
  }

  DEFINE_ACCEPT
};

TLANG_NAMESPACE_END
