#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/expr.h"

TLANG_NAMESPACE_BEGIN

class FrontendAllocaStmt : public Stmt {
 public:
  Identifier ident;

  FrontendAllocaStmt(const Identifier &lhs, DataType type) : ident(lhs) {
    ret_type = VectorType(1, type);
  }

  DEFINE_ACCEPT
};

// For return values
class FrontendArgStoreStmt : public Stmt {
 public:
  int arg_id;
  Expr expr;

  FrontendArgStoreStmt(int arg_id, const Expr &expr)
      : arg_id(arg_id), expr(expr) {
  }

  // Arguments are considered global (nonlocal)
  virtual bool has_global_side_effect() const override {
    return true;
  }

  DEFINE_ACCEPT
};

class FrontendSNodeOpStmt : public Stmt {
 public:
  SNodeOpType op_type;
  SNode *snode;
  ExprGroup indices;
  Expr val;

  FrontendSNodeOpStmt(SNodeOpType op_type,
                      SNode *snode,
                      const ExprGroup &indices,
                      const Expr &val = Expr(nullptr));

  DEFINE_ACCEPT
};

class FrontendAssertStmt : public Stmt {
 public:
  std::string text;
  Expr val;

  FrontendAssertStmt(const std::string &text, const Expr &val)
      : text(text), val(val) {
  }

  DEFINE_ACCEPT
};

class FrontendAssignStmt : public Stmt {
 public:
  Expr lhs, rhs;

  FrontendAssignStmt(const Expr &lhs, const Expr &rhs);

  DEFINE_ACCEPT
};


class FrontendIfStmt : public Stmt {
 public:
  Expr condition;
  std::unique_ptr<Block> true_statements, false_statements;

  FrontendIfStmt(const Expr &condition) : condition(load_if_ptr(condition)) {
  }

  bool is_container_statement() const override {
    return true;
  }

  DEFINE_ACCEPT
};

class FrontendPrintStmt : public Stmt {
 public:
  Expr expr;
  std::string str;

  FrontendPrintStmt(const Expr &expr, const std::string &str)
      : expr(load_if_ptr(expr)), str(str) {
  }

  DEFINE_ACCEPT
};

class FrontendEvalStmt : public Stmt {
 public:
  Expr expr;
  Expr eval_expr;

  FrontendEvalStmt(const Expr &expr) : expr(load_if_ptr(expr)) {
  }

  DEFINE_ACCEPT
};

class FrontendForStmt : public Stmt {
 public:
  Expr begin, end;
  Expr global_var;
  std::unique_ptr<Block> body;
  std::vector<Identifier> loop_var_id;
  int vectorize;
  int parallelize;
  bool strictly_serialized;
  ScratchPadOptions scratch_opt;
  int block_dim;

  bool is_ranged() const {
    if (global_var.expr == nullptr) {
      return true;
    } else {
      return false;
    }
  }

  FrontendForStmt(const ExprGroup &loop_var, const Expr &global_var);

  FrontendForStmt(const Expr &loop_var, const Expr &begin, const Expr &end);

  bool is_container_statement() const override {
    return true;
  }

  DEFINE_ACCEPT
};

class FrontendFuncDefStmt : public Stmt {
 public:
  std::string funcid;
  std::unique_ptr<Block> body;

  FrontendFuncDefStmt(const std::string &funcid) : funcid(funcid) {
  }

  bool is_container_statement() const override {
    return true;
  }

  DEFINE_ACCEPT
};


class FrontendBreakStmt : public Stmt {
 public:
  FrontendBreakStmt() {
  }

  bool is_container_statement() const override {
    return false;
  }

  DEFINE_ACCEPT
};

class FrontendContinueStmt : public Stmt {
 public:
  FrontendContinueStmt() = default;

  bool is_container_statement() const override {
    return false;
  }

  DEFINE_ACCEPT
};

class FrontendWhileStmt : public Stmt {
 public:
  Expr cond;
  std::unique_ptr<Block> body;

  FrontendWhileStmt(const Expr &cond) : cond(load_if_ptr(cond)) {
  }

  bool is_container_statement() const override {
    return true;
  }

  DEFINE_ACCEPT
};



TLANG_NAMESPACE_END
