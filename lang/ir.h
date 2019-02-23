#pragma once
#include "util.h"
#include <taichi/util.h>
#include <taichi/testing.h>
#include "structural_node.h"

TLANG_NAMESPACE_BEGIN

// No Expr nodes - make everything as close to SSA as possible

class IRBuilder;
class IRNode;
class Block;
class Statement;
using Stmt = Statement;

// statements
class ConstStmt;
class IfStmt;

// frontend stmts
class FrontendIfStmt;
class FrontendForStmt;
class FrontendPrintStmt;
class RangeForStmt;
class FrontendWhileStmt;
class WhileStmt;
class WhileControlStmt;
class AssignStmt;
class AllocaStmt;
class UnaryOpStmt;
class BinaryOpStmt;
class LocalLoadStmt;
class LocalStoreStmt;
class GlobalPtrStmt;
class GlobalLoadStmt;
class GlobalStoreStmt;
class PrintStmt;
class WhileControlStmt;

class SNode;

namespace irpass {

void print(IRNode *root);
void lower(IRNode *root);
void typecheck(IRNode *root);
void loop_vectorize(IRNode *root);
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

  std::string str() const {
    return fmt::format("{}x{}", data_type_name(data_type), width);
  }
};

class DecoratorRecorder {
 public:
  int vectorize;

  DecoratorRecorder() {
    reset();
  }

  void reset() {
    vectorize = -1;
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

  void create_function() {
  }
};

inline IRBuilder &current_ast_builder() {
  return context->builder();
}

class ExpressionHandle;
using ExprH = ExpressionHandle;

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

using VecStatement = std::vector<std::unique_ptr<Statement>>;

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
  DEFINE_VISIT(AssignStmt);
  DEFINE_VISIT(AllocaStmt);
  DEFINE_VISIT(BinaryOpStmt);
  DEFINE_VISIT(UnaryOpStmt);
  DEFINE_VISIT(LocalLoadStmt);
  DEFINE_VISIT(LocalStoreStmt);
  DEFINE_VISIT(GlobalLoadStmt);
  DEFINE_VISIT(GlobalStoreStmt);
  DEFINE_VISIT(GlobalPtrStmt);
  DEFINE_VISIT(IfStmt);
  DEFINE_VISIT(FrontendIfStmt);
  DEFINE_VISIT(PrintStmt);
  DEFINE_VISIT(FrontendPrintStmt);
  DEFINE_VISIT(ConstStmt);
  DEFINE_VISIT(FrontendForStmt);
  DEFINE_VISIT(RangeForStmt);
  DEFINE_VISIT(FrontendWhileStmt);
  DEFINE_VISIT(WhileStmt);
  DEFINE_VISIT(WhileControlStmt);
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

struct StmtAttribute {
  int vector_width;
};

template <typename T>
struct LaneAttribute {
  std::vector<T> data;

  LaneAttribute() {
    data.resize(1, T());
  }

  LaneAttribute(const T &t) {
    data.resize(1);
    data[0] = t;
  }

  void resize(int s) {
    data.resize();
  }

  std::size_t size() {
    return data.size();
  }

  T &operator[](int i) {
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
};

class Statement : public IRNode {
 public:
  static int id_counter;
  int id;
  Block *parent;
  std::vector<Stmt **> operands;

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

  void insert_before(std::unique_ptr<Stmt> &&new_stmt);

  void insert_after(std::unique_ptr<Stmt> &&new_stmt);
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

class ExpressionGroup;

class ExpressionHandle {
 public:
  std::shared_ptr<Expression> expr;

  ExpressionHandle() {
  }

  ExpressionHandle(int32 x);

  ExpressionHandle(float32 x);

  ExpressionHandle(std::shared_ptr<Expression> expr) : expr(expr) {
  }

  ExpressionHandle(Identifier id);

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

  void operator=(const ExpressionHandle &o);

  ExpressionHandle operator[](ExpressionGroup);

  void set(const ExpressionHandle &o) {
    expr = o.expr;
  }

  std::string serialize() const {
    TC_ASSERT(expr);
    return expr->serialize();
  }

  void *evaluate_addr(int i, int j, int k, int l);

  template <typename... Indices>
  void *val_tmp(Indices... indices);

  template <typename T, typename... Indices>
  T &val(Indices... indices);
};

class ExpressionGroup {
 public:
  std::vector<ExprH> exprs;

  ExpressionGroup() {
  }

  ExpressionGroup(ExprH a) {
    exprs.push_back(a);
  }

  ExpressionGroup(ExprH a, ExprH b) {
    exprs.push_back(a);
    exprs.push_back(b);
  }

  ExpressionGroup(ExpressionGroup a, const ExprH &b) {
    exprs = a.exprs;
    exprs.push_back(b);
  }

  std::size_t size() const {
    return exprs.size();
  }
};

inline ExpressionGroup operator,(const ExprH &a, const ExprH &b) {
  return ExpressionGroup(a, b);
}

inline ExpressionGroup operator,(const ExpressionGroup &a, const ExprH &b) {
  return ExpressionGroup(a, b);
}

// updates mask, break if no active
class WhileControlStmt : public Statement {
 public:
  Ident mask;
  Stmt *cond;
  WhileControlStmt(Ident mask, Stmt *cond) : mask(mask), cond(cond) {
  }
  DEFINE_ACCEPT;
};

class UnaryOpStmt : public Statement {
 public:
  UnaryType op_type;
  Statement *rhs;

  UnaryOpStmt(UnaryType op_type, Statement *rhs) : op_type(op_type), rhs(rhs) {
    add_operand(this->rhs);
  }

  DEFINE_ACCEPT
};

class UnaryOpExpression : public Expression {
 public:
  UnaryType type;
  ExpressionHandle rhs;

  UnaryOpExpression(UnaryType type, ExpressionHandle rhs)
      : type(type), rhs(rhs) {
  }

  std::string serialize() override {
    return fmt::format("({} {})", unary_type_name(type), rhs->serialize());
  }

  void flatten(VecStatement &ret) override {
    if (stmt)
      return;
    rhs->flatten(ret);
    ret.push_back(std::make_unique<UnaryOpStmt>(type, rhs->stmt));
    stmt = ret.back().get();
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
  ExpressionHandle lhs, rhs;

  BinaryOpExpression(BinaryType type,
                     ExpressionHandle lhs,
                     ExpressionHandle rhs)
      : type(type), lhs(lhs), rhs(rhs) {
  }

  std::string serialize() override {
    return fmt::format("({} {} {})", lhs->serialize(), binary_type_symbol(type),
                       rhs->serialize());
  }

  void flatten(VecStatement &ret) override {
    if (stmt)
      return;
    lhs->flatten(ret);
    rhs->flatten(ret);
    ret.push_back(std::make_unique<BinaryOpStmt>(type, lhs->stmt, rhs->stmt));
    stmt = ret.back().get();
  }
};

class GlobalPtrStmt : public Statement {
 public:
  SNode *snode;
  std::vector<Stmt *> indices;

  GlobalPtrStmt(SNode *snode, const std::vector<Stmt *> &indices)
      : snode(snode), indices(indices) {
    for (int i = 0; i < (int)indices.size(); i++) {
      add_operand(this->indices[i]);
    }
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
  ExprH var;
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
    if (stmt)
      return;
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
  inline ExpressionHandle operator op(ExpressionHandle lhs,                   \
                                      ExpressionHandle rhs) {                 \
    return ExpressionHandle(                                                  \
        std::make_shared<BinaryOpExpression>(BinaryType::op_name, lhs, rhs)); \
  }

inline ExprH operator-(ExprH expr) {
  return ExprH(std::make_shared<UnaryOpExpression>(UnaryType::neg, expr));
}

DEFINE_EXPRESSION_OP(+, add)
DEFINE_EXPRESSION_OP(-, sub)
DEFINE_EXPRESSION_OP(*, mul)
DEFINE_EXPRESSION_OP(/, div)
DEFINE_EXPRESSION_OP(%, mod)
DEFINE_EXPRESSION_OP(<, cmp_lt)
DEFINE_EXPRESSION_OP(<=, cmp_le)
DEFINE_EXPRESSION_OP(>, cmp_gt)
DEFINE_EXPRESSION_OP(>=, cmp_ge)
DEFINE_EXPRESSION_OP(==, cmp_eq)

class Block : public IRNode {
 public:
  Block *parent;
  std::vector<std::unique_ptr<Statement>> statements;
  std::map<Ident, VectorType> local_variables;
  Ident *inner_loop_variable;

  Block() {
    inner_loop_variable = nullptr;
    parent = nullptr;
  }

  void insert(std::unique_ptr<Statement> &&stmt, int location = -1) {
    stmt->parent = this;
    if (location == -1) {
      statements.push_back(std::move(stmt));
    } else {
      statements.insert(statements.begin() + location, std::move(stmt));
    }
  }

  void replace_with(Statement *old_statement,
                    std::unique_ptr<Statement> &&new_statement) {
    std::vector<std::unique_ptr<Statement>> vec;
    vec.push_back(std::move(new_statement));
    replace_with(old_statement, vec);
  }

  void replace_with(Statement *old_statement,
                    std::vector<std::unique_ptr<Statement>> &new_statements) {
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

  VectorType lookup_var(Ident ident) const {
    auto ptr = local_variables.find(ident);
    if (ptr != local_variables.end()) {
      return ptr->second;
    } else {
      if (parent) {
        return parent->lookup_var(ident);
      } else {
        return VectorType(1, DataType::unknown);
      }
    }
  }

  DEFINE_ACCEPT
};

class AssignStmt : public Statement {
 public:
  ExprH lhs, rhs;

  AssignStmt(ExprH lhs, ExprH rhs);

  DEFINE_ACCEPT
};

class AllocaStmt : public Statement {
 public:
  Ident ident;

  AllocaStmt(Ident lhs, DataType type) : ident(lhs) {
    ret_type = VectorType(1, type);
  }
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

class LocalLoadStmt : public Statement {
 public:
  LaneAttribute<Ident> ident;

  LocalLoadStmt(Ident ident) : ident(ident) {
  }

  DEFINE_ACCEPT;
};

class LocalStoreStmt : public Statement {
 public:
  Ident ident;
  Statement *stmt;

  LocalStoreStmt(Ident ident, Statement *stmt) : ident(ident), stmt(stmt) {
    add_operand(this->stmt);
  }

  DEFINE_ACCEPT;
};

class IfStmt : public Statement {
 public:
  Statement *cond;
  std::unique_ptr<Block> true_statements, false_statements;

  IfStmt(Statement *cond) : cond(cond) {
    add_operand(this->cond);
  }

  DEFINE_ACCEPT
};

class FrontendIfStmt : public Statement {
 public:
  ExpressionHandle condition;
  std::unique_ptr<Block> true_statements, false_statements;

  FrontendIfStmt(ExpressionHandle condition) : condition(condition) {
  }

  DEFINE_ACCEPT
};

class FrontendPrintStmt : public Statement {
 public:
  ExprH expr;

  FrontendPrintStmt(ExprH expr) : expr(expr) {
  }

  DEFINE_ACCEPT
};

class PrintStmt : public Statement {
 public:
  Statement *stmt;

  PrintStmt(Statement *stmt) : stmt(stmt) {
    add_operand(this->stmt);
  }

  DEFINE_ACCEPT
};

class If {
 public:
  FrontendIfStmt *stmt;

  If(ExpressionHandle cond) {
    auto stmt_tmp = std::make_unique<FrontendIfStmt>(cond);
    stmt = stmt_tmp.get();
    current_ast_builder().insert(std::move(stmt_tmp));
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
    value = x;
  }

  ConstStmt(float32 x) {
    ret_type = VectorType(1, DataType::f32);
    value = x;
  }

  void repeat(int factor) override {
    Statement::repeat(factor);
    value.repeat(factor);
  }

  DEFINE_ACCEPT
};

class FrontendForStmt : public Statement {
 public:
  ExprH begin, end;
  std::unique_ptr<Block> body;
  Ident loop_var_id;
  int vectorize;

  FrontendForStmt(ExprH loop_var, ExprH begin, ExprH end);

  DEFINE_ACCEPT
};

// General range for
class RangeForStmt : public Statement {
 public:
  Ident loop_var;
  Statement *begin, *end;
  std::unique_ptr<Block> body;
  int vectorize;

  RangeForStmt(Ident loop_var,
               Statement *begin,
               Statement *end,
               std::unique_ptr<Block> &&body,
               int vectorize)
      : loop_var(loop_var),
        begin(begin),
        end(end),
        body(std::move(body)),
        vectorize(vectorize) {
    add_operand(this->begin);
    add_operand(this->end);
  }

  DEFINE_ACCEPT
};

class WhileStmt : public Statement {
 public:
  Ident mask;
  std::unique_ptr<Block> body;

  WhileStmt(std::unique_ptr<Block> &&body) : body(std::move(body)) {
  }

  DEFINE_ACCEPT
};

class FrontendWhileStmt : public Statement {
 public:
  ExprH cond;
  std::unique_ptr<Block> body;

  FrontendWhileStmt(ExprH cond) : cond(cond) {
  }

  DEFINE_ACCEPT
};

inline void IRBuilder::insert(std::unique_ptr<Statement> &&stmt, int location) {
  TC_ASSERT(!stack.empty());
  stack.back()->insert(std::move(stmt), location);
}

inline void Print(const ExpressionHandle &a) {
  current_ast_builder().insert(std::make_unique<FrontendPrintStmt>(a));
}

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
    if (stmt)
      return;
    ret.push_back(std::make_unique<LocalLoadStmt>(id));
    stmt = ret.back().get();
  }
};

class GlobalLoadExpression : public Expression {
 public:
  ExprH ptr;
  GlobalLoadExpression(ExprH ptr) : ptr(ptr) {
  }

  std::string serialize() override {
    return "load ";
  }

  void flatten(VecStatement &ret) override {
    if (stmt)
      return;
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
    if (stmt)
      return;
    if (dt == DataType::f32) {
      ret.push_back(std::make_unique<ConstStmt>((float32)val));
    } else {
      ret.push_back(std::make_unique<ConstStmt>((int32)val));
    }
    stmt = ret.back().get();
  }
};

template <typename T>
inline void declare_var(ExpressionHandle &a) {
  current_ast_builder().insert(std::make_unique<AllocaStmt>(
      std::static_pointer_cast<IdExpression>(a.expr)->id, get_data_type<T>()));
}

inline ExprH ExpressionHandle::operator[](ExpressionGroup indices) {
  TC_ASSERT(is<GlobalVariableExpression>());
  return ExprH(std::make_shared<GlobalPtrExpression>(
      cast<GlobalVariableExpression>(), indices));
}

#define declare(x) \
  auto x = ExpressionHandle(std::make_shared<IdExpression>(#x));

#define var(type, x) declare_var<type>(x);

inline ExprH global_new(ExprH id_expr, DataType dt) {
  TC_ASSERT(id_expr.is<IdExpression>());
  auto ret = ExprH(std::make_shared<GlobalVariableExpression>(
      dt, id_expr.cast<IdExpression>()->id));
  return ret;
}

template <typename T, typename... Indices>
T &ExprH::val(Indices... indices) {
  auto e = this->cast<GlobalVariableExpression>();
  TC_ASSERT(is<GlobalVariableExpression>());

  if (get_data_type<T>() != e->snode->dt) {
    TC_ERROR("Cannot access type {} as type {}", data_type_name(e->snode->dt),
             data_type_name(get_data_type<T>()));
  }
  return *(T *)val_tmp(indices...);
}

inline ExprH load(ExprH ptr) {
  TC_ASSERT(ptr.is<GlobalPtrStmt>());
  return ExpressionHandle(std::make_shared<GlobalLoadExpression>(ptr));
}

extern DecoratorRecorder dec;

inline void Vectorize(int v) {
  dec.vectorize = v;
}

class For {
 public:
  For(ExprH i, ExprH s, ExprH e, const std::function<void()> &func) {
    auto stmt_unique = std::make_unique<FrontendForStmt>(i, s, e);
    auto stmt = stmt_unique.get();
    current_ast_builder().insert(std::move(stmt_unique));
    auto _ = current_ast_builder().create_scope(stmt->body);
    func();
  }
};

class While {
 public:
  While(ExprH cond, const std::function<void()> &func) {
    auto while_stmt = std::make_unique<FrontendWhileStmt>(cond);
    FrontendWhileStmt *ptr = while_stmt.get();
    current_ast_builder().insert(std::move(while_stmt));
    auto _ = current_ast_builder().create_scope(ptr->body);
    func();
  }
};

TLANG_NAMESPACE_END
