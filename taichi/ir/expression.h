#pragma once

#include "taichi/util/str.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/expr.h"

TLANG_NAMESPACE_BEGIN

#include "taichi/ir/expression_ops.h"

// always a tree - used as rvalues
class Expression {
 public:
  Stmt *stmt;
  std::string tb;
  std::map<std::string, std::string> attributes;
  DataType ret_type;

  struct FlattenContext {
    VecStatement stmts;
    Block *current_block = nullptr;

    inline Stmt *push_back(pStmt &&stmt) {
      return stmts.push_back(std::move(stmt));
    }

    template <typename T, typename... Args>
    T *push_back(Args &&... args) {
      return stmts.push_back<T>(std::forward<Args>(args)...);
    }

    Stmt *back_stmt() {
      return stmts.back().get();
    }
  };

  Expression() {
    stmt = nullptr;
  }

  virtual void type_check() {
    // TODO: make it pure virtual after type_check for all expressions are
    // implemented
  }

  virtual void serialize(std::ostream &ss) = 0;

  virtual void flatten(FlattenContext *ctx) {
    TI_NOT_IMPLEMENTED;
  };

  virtual bool is_lvalue() const {
    return false;
  }

  virtual ~Expression() {
  }

  void set_attribute(const std::string &key, const std::string &value) {
    attributes[key] = value;
  }

  std::string get_attribute(const std::string &key) const;
};

class ExprGroup {
 public:
  std::vector<Expr> exprs;

  ExprGroup() {
  }

  ExprGroup(const Expr &a) {
    exprs.emplace_back(a);
  }

  ExprGroup(const Expr &a, const Expr &b) {
    exprs.emplace_back(a);
    exprs.emplace_back(b);
  }

  ExprGroup(const ExprGroup &a, const Expr &b) {
    exprs.resize(a.size() + 1);

    for (int i = 0; i < a.size(); ++i) {
      exprs[i].set_or_insert_assignment(a.exprs[i]);
    }
    exprs.back().set_or_insert_assignment(b);
  }

  ExprGroup(const Expr &a, const ExprGroup &b) {
    exprs.resize(b.size() + 1);
    exprs.front().set_or_insert_assignment(a);
    for (int i = 0; i < b.size(); i++) {
      exprs[i + 1].set_or_insert_assignment(b.exprs[i]);
    }
  }

  void push_back(const Expr &expr) {
    exprs.emplace_back(expr);
  }

  std::size_t size() const {
    return exprs.size();
  }

  const Expr &operator[](int i) const {
    return exprs[i];
  }

  Expr &operator[](int i) {
    return exprs[i];
  }

  void serialize(std::ostream &ss) const;

  std::string serialize() const;
};

inline ExprGroup operator,(const Expr &a, const Expr &b) {
  return ExprGroup(a, b);
}

inline ExprGroup operator,(const ExprGroup &a, const Expr &b) {
  return ExprGroup(a, b);
}

TLANG_NAMESPACE_END
