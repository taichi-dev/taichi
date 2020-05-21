#pragma once

#include "taichi/ir/expr.h"

TLANG_NAMESPACE_BEGIN

#include "taichi/ir/expression_ops.h"

// always a tree - used as rvalues
class Expression {
 public:
  Stmt *stmt;
  std::string tb;
  std::map<std::string, std::string> attributes;

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

  virtual std::string serialize() = 0;

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
    exprs.push_back(a);
  }

  ExprGroup(const Expr &a, const Expr &b) {
    exprs.push_back(a);
    exprs.push_back(b);
  }

  ExprGroup(const ExprGroup &a, const Expr &b) {
    exprs = a.exprs;
    exprs.push_back(b);
  }

  ExprGroup(const Expr &a, const ExprGroup &b) {
    exprs = b.exprs;
    exprs.insert(exprs.begin(), a);
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

  std::string serialize() const;
  ExprGroup loaded() const;
};

inline ExprGroup operator,(const Expr &a, const Expr &b) {
  return ExprGroup(a, b);
}

inline ExprGroup operator,(const ExprGroup &a, const Expr &b) {
  return ExprGroup(a, b);
}

TLANG_NAMESPACE_END
