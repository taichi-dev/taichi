#pragma once
#include "util.h"
#include <taichi/util.h>
#include <taichi/testing.h>
#include "structural_node.h"

TLANG_NAMESPACE_BEGIN

class IRBuilder;
class IRNode;
class Block;
class Statement;
using Stmt = Statement;
using pStmt = std::unique_ptr<Statement>;

class SNode;
class Expression;
class Expr;
class ExpressionGroup;

// Frontend Statements
class FrontendIfStmt;
class FrontendForStmt;
class FrontendPrintStmt;
class FrontendWhileStmt;
class FrontendAllocaStmt;
class FrontendAssignStmt;

// Midend Statement
class ConstStmt;

// Without per-lane attributes:
class RangeForStmt;
class IfStmt;
class WhileStmt;
class WhileControlStmt;
class UnaryOpStmt;
class BinaryOpStmt;
class AllocaStmt;
class PrintStmt;
class RandStmt;

// With per-lane attributes:
class GlobalLoadStmt;
class GlobalStoreStmt;
class LocalLoadStmt;
class LocalStoreStmt;
class GlobalPtrStmt;

// IR passes
namespace irpass {

void print(IRNode *root);
void lower(IRNode *root);
void typecheck(IRNode *root);
void loop_vectorize(IRNode *root);
void slp_vectorize(IRNode *root);
void replace_all_usages_with(IRNode *root, Stmt *old_stmt, Stmt *new_stmt);

}  // namespace irpass

struct VectorType {
  int width;
  DataType data_type;

  VectorType(int width, DataType data_type)
      : width(width), data_type(data_type) {
  }

  VectorType() : width(1), data_type(DataType::unknown) {
  }

  bool operator==(const VectorType &o) const {
    return width == o.width && data_type == o.data_type;
  }

  bool operator!=(const VectorType &o) const {
    return !(*this == o);
  }

  std::string str() const {
    return fmt::format("{}x{}", data_type_name(data_type), width);
  }
};

class DecoratorRecorder {
 public:
  int vectorize;
  int parallelize;

  DecoratorRecorder() {
    reset();
  }

  void reset() {
    vectorize = -1;
    parallelize = 0;
  }
};

class FrontendContext {
 private:
  std::unique_ptr<IRBuilder> current_builder;
  std::unique_ptr<Block> root_node;

 public:
  FrontendContext();

  IRBuilder &builder() {
    return *current_builder;
  }

  IRNode *root();
};

extern std::unique_ptr<FrontendContext> context;

class IRBuilder {
 private:
  std::vector<Block *> stack;

 public:
  IRBuilder(Block *initial) {
    stack.push_back(initial);
  }

  void insert(std::unique_ptr<Statement> &&stmt, int location = -1);

  struct ScopeGuard {
    IRBuilder *builder;
    Block *list;
    ScopeGuard(IRBuilder *builder, Block *list) : builder(builder), list(list) {
      builder->stack.push_back(list);
    }

    ~ScopeGuard() {
      builder->stack.pop_back();
    }
  };

  ScopeGuard create_scope(std::unique_ptr<Block> &list);

  Block *current_block() {
    if (stack.empty())
      return nullptr;
    else
      return stack.back();
  }

  void create_function() {
  }
};

inline IRBuilder &current_ast_builder() {
  return context->builder();
}

inline Expr load_if_ptr(const Expr &ptr);

class Identifier {
 public:
  static int id_counter;
  std::string name_;

  int id;

  Identifier(std::string name_ = "") : name_(name_) {
    id = id_counter++;
  }

  std::string raw_name() const {
    if (name_.empty())
      return fmt::format("tmp{}", id);
    else
      return name_;
  }

  std::string name() const {
    return "@" + raw_name();
  }

  bool operator<(const Identifier &o) const {
    return id < o.id;
  }

  bool operator==(const Identifier &o) const {
    return id == o.id;
  }
};

using Ident = Identifier;

class VecStatement {
 public:
  std::vector<pStmt> stmts;

  VecStatement() {
  }

  void push_back(pStmt &&stmt) {
    stmts.push_back(std::move(stmt));
  }

  template <typename T, typename... Args>
  T *push_back(Args &&... args) {
    auto up = std::make_unique<T>(std::forward<Args>(args)...);
    auto ptr = up.get();
    stmts.push_back(std::move(up));
    return ptr;
  }

  pStmt &back() {
    return stmts.back();
  }

  std::size_t size() const {
    return stmts.size();
  }

  pStmt &operator[](int i) {
    return stmts[i];
  }
};

class IRVisitor {
 public:
  bool allow_undefined_visitor;
  bool invoke_default_visitor;

  IRVisitor() {
    allow_undefined_visitor = false;
    invoke_default_visitor = false;
  }

  // default visitor
  virtual void visit(Statement *stmt) {
    if (!allow_undefined_visitor) {
      TC_ERROR(
          "missing visitor function. Is the statement class registered via "
          "DEFINE_VISIT?");
    }
  }

#define DEFINE_VISIT(T)            \
  virtual void visit(T *stmt) {    \
    if (allow_undefined_visitor) { \
      if (invoke_default_visitor)  \
        visit((Stmt *)stmt);       \
    } else                         \
      TC_NOT_IMPLEMENTED;          \
  }

  DEFINE_VISIT(Block);

  DEFINE_VISIT(FrontendIfStmt);
  DEFINE_VISIT(FrontendAssignStmt);
  DEFINE_VISIT(FrontendAllocaStmt);
  DEFINE_VISIT(FrontendPrintStmt);
  DEFINE_VISIT(FrontendForStmt);
  DEFINE_VISIT(FrontendWhileStmt);

  DEFINE_VISIT(AllocaStmt);
  DEFINE_VISIT(BinaryOpStmt);
  DEFINE_VISIT(UnaryOpStmt);
  DEFINE_VISIT(LocalLoadStmt);
  DEFINE_VISIT(LocalStoreStmt);
  DEFINE_VISIT(GlobalLoadStmt);
  DEFINE_VISIT(GlobalStoreStmt);
  DEFINE_VISIT(GlobalPtrStmt);
  DEFINE_VISIT(IfStmt);
  DEFINE_VISIT(PrintStmt);
  DEFINE_VISIT(ConstStmt);
  DEFINE_VISIT(RangeForStmt);
  DEFINE_VISIT(WhileStmt);
  DEFINE_VISIT(WhileControlStmt);
  DEFINE_VISIT(RandStmt);
};

class IRNode {
 public:
  virtual void accept(IRVisitor *visitor) {
    TC_NOT_IMPLEMENTED
  }
};

#define DEFINE_ACCEPT               \
  void accept(IRVisitor *visitor) { \
    visitor->visit(this);           \
  }

template <typename T>
struct LaneAttribute {
  std::vector<T> data;

  LaneAttribute() {
  }

  LaneAttribute(const T &t) {
    data.resize(1);
    data[0] = t;
  }

  void resize(int s) {
    data.resize(s);
  }

  void push_back(const T &t) {
    data.push_back(t);
  }

  std::size_t size() const {
    return data.size();
  }

  T &operator[](int i) {
    return data[i];
  }

  const T &operator[](int i) const {
    return data[i];
  }

  // for initializing single lane
  void operator=(const T &t) {
    TC_ASSERT(data.size() == 1);
    data[0] = t;
  }

  void repeat(int factor) {
    std::vector<T> new_data;
    for (int i = 0; i < factor; i++) {
      for (int j = 0; j < (int)data.size(); j++) {
        new_data.push_back(data[j]);
      }
    }
    data = new_data;
  }

  std::string serialize(std::function<std::string(const T &t)> func,
                        std::string bracket = "") {
    std::string ret = bracket;
    for (int i = 0; i < (int)data.size(); i++) {
      ret += func(data[i]);
      if (i + 1 < (int)data.size()) {
        ret += ", ";
      }
    }
    if (bracket == "<") {
      ret += ">";
    } else if (bracket == "{") {
      ret += "}";
    } else if (bracket == "(") {
      ret += ")";
    } else if (bracket != "") {
      TC_P(bracket);
      TC_NOT_IMPLEMENTED
    }
    return ret;
  }

  std::string serialize(std::string bracket = "") {
    std::string ret = bracket;
    for (int i = 0; i < (int)data.size(); i++) {
      ret += fmt::format("{}", data[i]);
      if (i + 1 < (int)data.size()) {
        ret += ", ";
      }
    }
    if (bracket == "<") {
      ret += ">";
    } else if (bracket == "{") {
      ret += "}";
    } else if (bracket == "(") {
      ret += ")";
    } else if (bracket != "") {
      TC_P(bracket);
      TC_NOT_IMPLEMENTED
    }
    return ret;
  }

  operator T() const {
    TC_ASSERT(data.size() == 1);
    return data[0];
  }

  LaneAttribute &operator+=(const LaneAttribute &o) {
    for (int i = 0; i < (int)o.size(); i++) {
      push_back(o[i]);
    }
    return *this;
  }
};

class Statement : public IRNode {
 public:
  static int id_counter;
  int id;
  Block *parent;
  std::vector<Stmt **> operands;

  int &width() {
    return ret_type.width;
  }

  VectorType ret_type;

  Statement(const Statement &stmt) = delete;

  Statement() {
    parent = nullptr;
    id = id_counter++;
  }

  std::string ret_data_type_name() const {
    return ret_type.str();
  }

  std::string type_hint() const {
    if (ret_type.data_type == DataType::unknown)
      return "";
    else
      return fmt::format("<{}> ", ret_data_type_name());
  }

  std::string name() {
    return fmt::format("${}", id);
  }

  std::string raw_name() {
    return fmt::format("tmp{}", id);
  }

  template <typename T>
  bool is() const {
    return dynamic_cast<const T *>(this) != nullptr;
  }

  template <typename T>
  T *as() {
    TC_ASSERT(is<T>());
    return dynamic_cast<T *>(this);
  }

  int num_operands() const {
    return operands.size();
  }

  Statement *&operand(int i) {
    TC_ASSERT(0 <= i && i < (int)operands.size());
    return *operands[i];
  }

  void add_operand(Statement *&stmt) {
    operands.push_back(&stmt);
  }

  IRNode *get_ir_root();

  virtual void repeat(int factor) {
    ret_type.width *= factor;
  }

  void replace_with(Stmt *new_stmt) {
    auto root = get_ir_root();
    irpass::replace_all_usages_with(root, this, new_stmt);
    // Note: the current structure should have been destroyed now..
  }

  virtual void replace_operand_with(Stmt *old_stmt, Stmt *new_stmt) {
    for (int i = 0; i < num_operands(); i++) {
      if (operand(i) == old_stmt) {
        operand(i) = new_stmt;
      }
    }
  }

  void insert_before_me(std::unique_ptr<Stmt> &&new_stmt);

  void insert_after_me(std::unique_ptr<Stmt> &&new_stmt);

  virtual bool integral_operands() const {
    return true;
  }

  template <typename T, typename... Args>
  static pStmt make(Args &&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
  }
};

// always a tree - used as rvalues
class Expression {
 public:
  Stmt *stmt;
  Expression() {
    stmt = nullptr;
  }

  virtual std::string serialize() = 0;
  virtual void flatten(VecStatement &ret) {
    TC_NOT_IMPLEMENTED;
  };
};

class Expr {
 public:
  std::shared_ptr<Expression> expr;

  Expr() {
  }

  Expr(int32 x);

  Expr(float32 x);

  Expr(std::shared_ptr<Expression> expr) : expr(expr) {
  }

  void set(const Expr &o) {
    expr = o.expr;
  }

  Expr(const Expr &o) {
    set(o);
  }

  Expr(Expr &&o) {
    set(o);
  }

  Expr(Identifier id);

  Expression *operator->() {
    return expr.get();
  }

  template <typename T>
  Handle<T> cast() const {
    return std::dynamic_pointer_cast<T>(expr);
  }

  template <typename T>
  bool is() const {
    return cast<T>() != nullptr;
  }

  void operator=(const Expr &o);

  Expr operator[](ExpressionGroup);

  std::string serialize() const {
    TC_ASSERT(expr);
    return expr->serialize();
  }

  void *evaluate_addr(int i, int j, int k, int l);

  template <typename... Indices>
  void *val_tmp(DataType dt, Indices... indices);

  template <typename T, typename... Indices>
  T &val(Indices... indices);

  void operator+=(const Expr &o);
  void operator-=(const Expr &o);
  void operator*=(const Expr &o);
  void operator/=(const Expr &o);
};

class ExpressionGroup {
 public:
  std::vector<Expr> exprs;

  ExpressionGroup() {
  }

  ExpressionGroup(const Expr &a) {
    exprs.push_back(a);
  }

  ExpressionGroup(const Expr &a, const Expr &b) {
    exprs.push_back(a);
    exprs.push_back(b);
  }

  ExpressionGroup(ExpressionGroup a, const Expr &b) {
    exprs = a.exprs;
    exprs.push_back(b);
  }

  std::size_t size() const {
    return exprs.size();
  }
};

inline ExpressionGroup operator,(const Expr &a, const Expr &b) {
  return ExpressionGroup(a, b);
}

inline ExpressionGroup operator,(const ExpressionGroup &a, const Expr &b) {
  return ExpressionGroup(a, b);
}

class FrontendAllocaStmt : public Statement {
 public:
  Ident ident;

  FrontendAllocaStmt(Ident lhs, DataType type) : ident(lhs) {
    ret_type = VectorType(1, type);
  }

  DEFINE_ACCEPT
};

class AllocaStmt : public Statement {
 public:
  AllocaStmt(DataType type) {
    ret_type = VectorType(1, type);
  }

  DEFINE_ACCEPT
};

// updates mask, break if no active
class WhileControlStmt : public Statement {
 public:
  Stmt *mask;
  Stmt *cond;
  WhileControlStmt(Stmt *mask, Stmt *cond) : mask(mask), cond(cond) {
  }
  DEFINE_ACCEPT;
};

class UnaryOpStmt : public Statement {
 public:
  UnaryType op_type;
  Statement *rhs;
  DataType cast_type;

  UnaryOpStmt(UnaryType op_type, Statement *rhs) : op_type(op_type), rhs(rhs) {
    add_operand(this->rhs);
    cast_type = DataType::unknown;
  }

  DEFINE_ACCEPT
};

class RandStmt : public Statement {
 public:
  RandStmt(DataType dt) {
    ret_type.data_type = dt;
  }

  DEFINE_ACCEPT
};

class RandExpression : public Expression {
 public:
  DataType dt;

  RandExpression(DataType dt) : dt(dt) {
  }

  std::string serialize() override {
    return fmt::format("rand<{}>()", data_type_name(dt));
  }

  void flatten(VecStatement &ret) override {
    auto ran = std::make_unique<RandStmt>(dt);
    ret.push_back(std::move(ran));
    stmt = ret.back().get();
  }
};

class UnaryOpExpression : public Expression {
 public:
  UnaryType type;
  Expr rhs;
  DataType cast_type;

  UnaryOpExpression(UnaryType type, Expr rhs)
      : type(type), rhs(load_if_ptr(rhs)) {
    cast_type = DataType::unknown;
  }

  std::string serialize() override {
    if (type == UnaryType::cast) {
      return fmt::format("({}<{}> {})", unary_type_name(type),
                         data_type_name(cast_type), rhs->serialize());
    } else {
      return fmt::format("({} {})", unary_type_name(type), rhs->serialize());
    }
  }

  void flatten(VecStatement &ret) override {
    rhs->flatten(ret);
    auto unary = std::make_unique<UnaryOpStmt>(type, rhs->stmt);
    if (type == UnaryType::cast)
      unary->cast_type = cast_type;
    stmt = unary.get();
    ret.push_back(std::move(unary));
  }
};

class BinaryOpStmt : public Statement {
 public:
  BinaryType op_type;
  Statement *lhs, *rhs;

  BinaryOpStmt(BinaryType op_type, Statement *lhs, Statement *rhs)
      : op_type(op_type), lhs(lhs), rhs(rhs) {
    add_operand(this->lhs);
    add_operand(this->rhs);
  }

  DEFINE_ACCEPT
};

class BinaryOpExpression : public Expression {
 public:
  BinaryType type;
  Expr lhs, rhs;

  BinaryOpExpression(BinaryType type, const Expr &lhs, const Expr &rhs)
      : type(type) {
    this->lhs.set(load_if_ptr(lhs));
    this->rhs.set(load_if_ptr(rhs));
  }

  std::string serialize() override {
    return fmt::format("({} {} {})", lhs->serialize(), binary_type_symbol(type),
                       rhs->serialize());
  }

  void flatten(VecStatement &ret) override {
    // if (stmt)
    //  return;
    lhs->flatten(ret);
    rhs->flatten(ret);
    ret.push_back(std::make_unique<BinaryOpStmt>(type, lhs->stmt, rhs->stmt));
    stmt = ret.back().get();
  }
};

class GlobalPtrStmt : public Stmt {
 public:
  LaneAttribute<SNode *> snode;
  std::vector<Stmt *> indices;

  GlobalPtrStmt(const LaneAttribute<SNode *> &snode,
                const std::vector<Stmt *> &indices)
      : snode(snode), indices(indices) {
    TC_TAG;
    for (int i = 0; i < (int)snode.size(); i++) {
      TC_ASSERT(snode[i] != nullptr);
      TC_ASSERT(snode[0]->dt == snode[i]->dt);
    }
    TC_TAG;
    for (int i = 0; i < (int)indices.size(); i++) {
      add_operand(this->indices[i]);
    }
    TC_TAG;
  }

  DEFINE_ACCEPT
};

class GlobalVariableExpression : public Expression {
 public:
  Identifier ident;
  DataType dt;
  SNode *snode;

  GlobalVariableExpression(DataType dt, Ident ident) : ident(ident), dt(dt) {
    snode = nullptr;
  }

  std::string serialize() override {
    return "#" + ident.name();
  }

  void flatten(VecStatement &ret) override {
    TC_ERROR("This should not be invoked");
    // ret.push_back(std::make_unique<LocalLoadStmt>(id));
  }
};

class GlobalPtrExpression : public Expression {
 public:
  Expr var;
  ExpressionGroup indices;

  GlobalPtrExpression(Handle<Expression> var, ExpressionGroup indices)
      : var(var), indices(indices) {
  }

  std::string serialize() override {
    std::string s = fmt::format("{}[", var.serialize());
    for (int i = 0; i < (int)indices.size(); i++) {
      s += indices.exprs[i]->serialize();
      if (i + 1 < (int)indices.size())
        s += ", ";
    }
    s += "]";
    return s;
  }

  void flatten(VecStatement &ret) override {
    std::vector<Stmt *> index_stmts;
    for (int i = 0; i < (int)indices.size(); i++) {
      indices.exprs[i]->flatten(ret);
      index_stmts.push_back(indices.exprs[i]->stmt);
    }
    ret.push_back(std::make_unique<GlobalPtrStmt>(
        var.cast<GlobalVariableExpression>()->snode, index_stmts));
    stmt = ret.back().get();
  }
};

#define DEFINE_EXPRESSION_OP(op, op_name)                                     \
  inline Expr operator op(const Expr &lhs, const Expr &rhs) {                 \
    return Expr(                                                              \
        std::make_shared<BinaryOpExpression>(BinaryType::op_name, lhs, rhs)); \
  }

inline Expr operator-(Expr expr) {
  return Expr(std::make_shared<UnaryOpExpression>(UnaryType::neg, expr));
}

#define DEFINE_EXPRESSION_OP_UNARY(opname)                                     \
  inline Expr opname(Expr expr) {                                              \
    return Expr(std::make_shared<UnaryOpExpression>(UnaryType::opname, expr)); \
  }

DEFINE_EXPRESSION_OP_UNARY(sqrt)
DEFINE_EXPRESSION_OP_UNARY(floor)
DEFINE_EXPRESSION_OP_UNARY(abs)
DEFINE_EXPRESSION_OP_UNARY(sin)
DEFINE_EXPRESSION_OP_UNARY(cos)

DEFINE_EXPRESSION_OP(+, add)
DEFINE_EXPRESSION_OP(-, sub)
DEFINE_EXPRESSION_OP(*, mul)
DEFINE_EXPRESSION_OP(/, div)
DEFINE_EXPRESSION_OP(%, mod)
DEFINE_EXPRESSION_OP(&&, bit_and)
DEFINE_EXPRESSION_OP(||, bit_or)
DEFINE_EXPRESSION_OP(<, cmp_lt)
DEFINE_EXPRESSION_OP(<=, cmp_le)
DEFINE_EXPRESSION_OP(>, cmp_gt)
DEFINE_EXPRESSION_OP(>=, cmp_ge)
DEFINE_EXPRESSION_OP(==, cmp_eq)

#define DEFINE_EXPRESSION_FUNC(op_name)                                       \
  inline Expr op_name(const Expr &lhs, const Expr &rhs) {                     \
    return Expr(                                                              \
        std::make_shared<BinaryOpExpression>(BinaryType::op_name, lhs, rhs)); \
  }

DEFINE_EXPRESSION_FUNC(min);
DEFINE_EXPRESSION_FUNC(max);

template <typename T>
inline Expr cast(Expr input) {
  auto ret = std::make_shared<UnaryOpExpression>(UnaryType::cast, input);
  ret->cast_type = get_data_type<T>();
  return Expr(ret);
}

class Block : public IRNode {
 public:
  Block *parent;
  std::vector<std::unique_ptr<Statement>> statements;
  std::map<Ident, Stmt *> local_var_alloca;
  Stmt *mask_var;
  int slp;
  Stmt *inner_loop_variable;

  Block() {
    inner_loop_variable = nullptr;
    mask_var = nullptr;
    parent = nullptr;
    slp = 1;
  }

  void insert(std::unique_ptr<Statement> &&stmt, int location = -1) {
    stmt->parent = this;
    if (location == -1) {
      statements.push_back(std::move(stmt));
    } else {
      statements.insert(statements.begin() + location, std::move(stmt));
    }
  }

  void set_statements(VecStatement &&stmts) {
    statements.clear();
    for (int i = 0; i < (int)stmts.size(); i++) {
      insert(std::move(stmts[i]), i);
    }
  }

  void replace_with(Statement *old_statement,
                    std::unique_ptr<Statement> &&new_statement) {
    VecStatement vec;
    vec.push_back(std::move(new_statement));
    replace_with(old_statement, vec);
  }

  void replace_with(Statement *old_statement, VecStatement &new_statements) {
    int location = -1;
    for (int i = 0; i < (int)statements.size(); i++) {
      if (old_statement == statements[i].get()) {
        location = i;
        break;
      }
    }
    TC_ASSERT(location != -1);
    statements.erase(statements.begin() + location);
    for (int i = (int)new_statements.size() - 1; i >= 0; i--) {
      insert(std::move(new_statements[i]), location);
    }
  }

  Stmt *lookup_var(Ident ident) const {
    auto ptr = local_var_alloca.find(ident);
    if (ptr != local_var_alloca.end()) {
      return ptr->second;
    } else {
      if (parent) {
        return parent->lookup_var(ident);
      } else {
        return nullptr;
      }
    }
  }

  Stmt *mask() {
    if (mask_var)
      return mask_var;
    else if (parent == nullptr) {
      return nullptr;
    } else {
      return parent->mask();
    }
  }

  DEFINE_ACCEPT
};

class FrontendAssignStmt : public Statement {
 public:
  Expr lhs, rhs;

  FrontendAssignStmt(Expr lhs, Expr rhs);

  DEFINE_ACCEPT
};

class GlobalLoadStmt : public Statement {
 public:
  Stmt *ptr;

  GlobalLoadStmt(Stmt *ptr) : ptr(ptr) {
    add_operand(this->ptr);
  }

  DEFINE_ACCEPT;
};

class GlobalStoreStmt : public Statement {
 public:
  Stmt *ptr, *data;

  GlobalStoreStmt(Stmt *ptr, Stmt *data) : ptr(ptr), data(data) {
    add_operand(this->ptr);
    add_operand(this->data);
  }

  DEFINE_ACCEPT;
};

struct LocalAddress {
  Stmt *var;
  int offset;

  LocalAddress() : LocalAddress(nullptr, 0) {
  }

  LocalAddress(Stmt *var, int offset) : var(var), offset(offset) {
  }
};

template <typename T>
std::string to_string(const T &);

class LocalLoadStmt : public Statement {
 public:
  LaneAttribute<LocalAddress> ptr;

  LocalLoadStmt(LaneAttribute<LocalAddress> ptr) : ptr(ptr) {
    for (int i = 0; i < (int)ptr.size(); i++) {
      add_operand(this->ptr[i].var);
    }
  }

  bool same_source() const {
    for (int i = 1; i < (int)ptr.size(); i++) {
      if (ptr[i].var != ptr[0].var)
        return false;
    }
    return true;
  }

  bool integral_operands() const override {
    return false;
  }

  DEFINE_ACCEPT;
};

class LocalStoreStmt : public Statement {
 public:
  Stmt *ident;
  Stmt *stmt;

  LaneAttribute<Stmt *> data;

  LocalStoreStmt(Stmt *ident, Statement *stmt) : ident(ident), stmt(stmt) {
    add_operand(this->ident);
    add_operand(this->stmt);
  }

  DEFINE_ACCEPT;
};

class IfStmt : public Statement {
 public:
  Stmt *cond;
  Stmt *true_mask, *false_mask;
  std::unique_ptr<Block> true_statements, false_statements;

  IfStmt(Statement *cond) : cond(cond) {
    add_operand(this->cond);
  }

  DEFINE_ACCEPT
};

class FrontendIfStmt : public Statement {
 public:
  Expr condition;
  std::unique_ptr<Block> true_statements, false_statements;

  FrontendIfStmt(Expr condition) : condition(condition) {
  }

  DEFINE_ACCEPT
};

class FrontendPrintStmt : public Statement {
 public:
  Expr expr;
  std::string str;

  FrontendPrintStmt(Expr expr, std::string str) : expr(expr), str(str) {
  }

  DEFINE_ACCEPT
};

class PrintStmt : public Statement {
 public:
  Statement *stmt;
  std::string str;

  PrintStmt(Statement *stmt, std::string str) : stmt(stmt), str(str) {
    add_operand(this->stmt);
  }

  DEFINE_ACCEPT
};

class If {
 public:
  FrontendIfStmt *stmt;

  If(Expr cond) {
    auto stmt_tmp = std::make_unique<FrontendIfStmt>(cond);
    stmt = stmt_tmp.get();
    current_ast_builder().insert(std::move(stmt_tmp));
  }

  If(Expr cond, const std::function<void()> &func) : If(cond) {
    Then(func);
  }

  If &Then(const std::function<void()> &func) {
    auto _ = current_ast_builder().create_scope(stmt->true_statements);
    func();
    return *this;
  }

  If &Else(const std::function<void()> &func) {
    auto _ = current_ast_builder().create_scope(stmt->false_statements);
    func();
    return *this;
  }
};

class ConstStmt : public Statement {
 public:
  LaneAttribute<long double> value;

  ConstStmt(int32 x) {
    ret_type = VectorType(1, DataType::i32);
    value.push_back(x);
  }

  ConstStmt(float32 x) {
    ret_type = VectorType(1, DataType::f32);
    value.push_back(x);
  }

  void repeat(int factor) override {
    Statement::repeat(factor);
    value.repeat(factor);
  }

  DEFINE_ACCEPT
};

class FrontendForStmt : public Statement {
 public:
  Expr begin, end;
  std::unique_ptr<Block> body;
  Ident loop_var_id;
  int vectorize;
  int parallelize;

  FrontendForStmt(Expr loop_var, Expr begin, Expr end);

  DEFINE_ACCEPT
};

// General range for
class RangeForStmt : public Statement {
 public:
  Stmt *loop_var;
  Stmt *begin, *end;
  std::unique_ptr<Block> body;
  int vectorize;
  int parallelize;

  RangeForStmt(Stmt *loop_var,
               Statement *begin,
               Statement *end,
               std::unique_ptr<Block> &&body,
               int vectorize,
               int parallelize)
      : loop_var(loop_var),
        begin(begin),
        end(end),
        body(std::move(body)),
        vectorize(vectorize),
        parallelize(parallelize) {
    add_operand(this->begin);
    add_operand(this->end);
  }

  DEFINE_ACCEPT
};

class WhileStmt : public Statement {
 public:
  Stmt *mask;
  std::unique_ptr<Block> body;

  WhileStmt(std::unique_ptr<Block> &&body) : body(std::move(body)) {
  }

  DEFINE_ACCEPT
};

class FrontendWhileStmt : public Statement {
 public:
  Expr cond;
  std::unique_ptr<Block> body;

  FrontendWhileStmt(Expr cond) : cond(load_if_ptr(cond)) {
  }

  DEFINE_ACCEPT
};

inline void IRBuilder::insert(std::unique_ptr<Statement> &&stmt, int location) {
  TC_ASSERT(!stack.empty());
  stack.back()->insert(std::move(stmt), location);
}

#define Print(x) Print_(x, #x);

inline void Print_(const Expr &a, std::string str) {
  current_ast_builder().insert(std::make_unique<FrontendPrintStmt>(a, str));
}

// TODO: fix this hack...
// for current ast
extern Block *current_block;

class IdExpression : public Expression {
 public:
  Identifier id;
  IdExpression(std::string name = "") : id(name) {
  }
  IdExpression(Identifier id) : id(id) {
  }

  std::string serialize() override {
    return id.name();
  }

  void flatten(VecStatement &ret) override {
    ret.push_back(std::make_unique<LocalLoadStmt>(
        LocalAddress(current_block->lookup_var(id), 0)));
    stmt = ret.back().get();
  }
};

class GlobalLoadExpression : public Expression {
 public:
  Expr ptr;
  GlobalLoadExpression(Expr ptr) : ptr(ptr) {
  }

  std::string serialize() override {
    return "load ";
  }

  void flatten(VecStatement &ret) override {
    // if (stmt)
    // return;
    ptr->flatten(ret);
    ret.push_back(std::make_unique<GlobalLoadStmt>(ptr->stmt));
    stmt = ret.back().get();
  }
};

class ConstExpression : public Expression {
 public:
  long double val;
  DataType dt;

  ConstExpression(int val) : val(val) {
    dt = DataType::i32;
  }

  ConstExpression(float32 val) : val(val) {
    dt = DataType::f32;
  }

  std::string serialize() override {
    return fmt::format("{}", val);
  }

  void flatten(VecStatement &ret) override {
    // if (stmt)
    // return;
    if (dt == DataType::f32) {
      ret.push_back(Stmt::make<ConstStmt>((float32)val));
    } else {
      ret.push_back(Stmt::make<ConstStmt>((int32)val));
    }
    stmt = ret.back().get();
  }
};

template <typename T>
inline void declare_var(Expr &a) {
  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(a.expr)->id, get_data_type<T>()));
}

inline void declare_var(Expr &a) {
  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(a.expr)->id, DataType::unknown));
}

inline Expr Expr::operator[](ExpressionGroup indices) {
  TC_ASSERT(is<GlobalVariableExpression>());
  return Expr(std::make_shared<GlobalPtrExpression>(
      cast<GlobalVariableExpression>(), indices));
}

#define declare(x) auto x = Expr(std::make_shared<IdExpression>(#x));

#define var(type, x) declare_var<type>(x);

#define local(x)  \
  declare(x);     \
  declare_var(x); \
  x

inline Expr global_new(Expr id_expr, DataType dt) {
  TC_ASSERT(id_expr.is<IdExpression>());
  auto ret = Expr(std::make_shared<GlobalVariableExpression>(
      dt, id_expr.cast<IdExpression>()->id));
  return ret;
}

template <typename T, typename... Indices>
T &Expr::val(Indices... indices) {
  auto e = this->cast<GlobalVariableExpression>();
  TC_ASSERT(is<GlobalVariableExpression>());
  return *(T *)val_tmp(get_data_type<T>(), indices...);
}

inline Expr load(Expr ptr) {
  TC_ASSERT(ptr.is<GlobalPtrExpression>());
  return Expr(std::make_shared<GlobalLoadExpression>(ptr));
}

inline Expr load_if_ptr(const Expr &ptr) {
  if (ptr.is<GlobalPtrExpression>()) {
    return load(ptr);
  } else {
    return ptr;
  }
}

extern DecoratorRecorder dec;

inline void Vectorize(int v) {
  dec.vectorize = v;
}

inline void Parallelize(int v) {
  dec.parallelize = v;
}

inline void SLP(int v) {
  current_ast_builder().current_block()->slp = v;
}

class For {
 public:
  For(Expr i, Expr s, Expr e, const std::function<void()> &func) {
    auto stmt_unique = std::make_unique<FrontendForStmt>(i, s, e);
    auto stmt = stmt_unique.get();
    current_ast_builder().insert(std::move(stmt_unique));
    auto _ = current_ast_builder().create_scope(stmt->body);
    func();
  }
};

class While {
 public:
  While(Expr cond, const std::function<void()> &func) {
    auto while_stmt = std::make_unique<FrontendWhileStmt>(cond);
    FrontendWhileStmt *ptr = while_stmt.get();
    current_ast_builder().insert(std::move(while_stmt));
    auto _ = current_ast_builder().create_scope(ptr->body);
    func();
  }
};

template <typename T>
Expr Rand() {
  return Expr(std::make_shared<RandExpression>(get_data_type<T>()));
}

TLANG_NAMESPACE_END
