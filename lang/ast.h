#include "util.h"
#include <taichi/util.h>
#include <taichi/testing.h>

TLANG_NAMESPACE_BEGIN

// No Expr nodes - make everything as close to SSA as possible

class ASTBuilder;
class ASTNode;
class Statement;
class StatementList;
class ConstStatement;
class ForStatement;
class WhileStatement;

class FrontendContext {
 private:
  std::unique_ptr<ASTBuilder> current_builder;
  Handle<StatementList> root_node;

 public:
  FrontendContext();

  ASTBuilder &builder() {
    return *current_builder;
  }

  ASTNode &root() {
    return *std::static_pointer_cast<ASTNode>(root_node);
  }
};

FrontendContext context;

class ASTBuilder {
 private:
  std::vector<Handle<StatementList>> stack;

 public:
  ASTBuilder(const Handle<StatementList> &initial) {
    stack.push_back(initial);
  }

  void insert(const Handle<Statement> &stmt);

  struct ScopeGuard {
    ASTBuilder *builder;
    Handle<StatementList> list;
    ScopeGuard(ASTBuilder *builder, const Handle<StatementList> &list)
        : builder(builder), list(list) {
      builder->stack.push_back(list);
    }

    ~ScopeGuard() {
      builder->stack.pop_back();
    }
  };

  ScopeGuard create_scope(Handle<StatementList> &list) {
    TC_ASSERT(list == nullptr);
    list = std::make_shared<StatementList>();
    return ScopeGuard(this, list);
  }

  void create_function() {
  }
};

ASTBuilder &current_ast_builder() {
  return context.builder();
}

class ExpressionHandle;
using ExprH = ExpressionHandle;

class Identifier {
 public:
  static int id_counter;

  int id;

  Identifier() {
    id = id_counter++;
  }

  void operator=(const ExpressionHandle &o);

  std::string name() {
    return fmt::format("id_{}", id);
  }
};

int Identifier::id_counter = 0;

using Id = Identifier;
class AssignmentStatement;
class AllocaStatement;
class BinaryOpStatement;
class UnaryOpStatement;
class IfStatement;
class PrintStatement;

class ASTVisitor {
 public:
#define DEFINE_VISIT(T)         \
  virtual void visit(T &stmt) { \
    TC_NOT_IMPLEMENTED;         \
  }

  DEFINE_VISIT(StatementList);
  DEFINE_VISIT(AssignmentStatement);
  DEFINE_VISIT(AllocaStatement);
  DEFINE_VISIT(BinaryOpStatement);
  DEFINE_VISIT(UnaryOpStatement);
  DEFINE_VISIT(IfStatement);
  DEFINE_VISIT(PrintStatement);
  DEFINE_VISIT(ConstStatement);
  DEFINE_VISIT(ForStatement);
  DEFINE_VISIT(WhileStatement);
};

class ASTNode {
 public:
  virtual void accept(ASTVisitor &visitor) {
    TC_NOT_IMPLEMENTED
  }
};

#define DEFINE_ACCEPT                \
  void accept(ASTVisitor &visitor) { \
    visitor.visit(*this);            \
  }

class Statement : public ASTNode {};

// always a tree - used as rvalues
class Expression {
 public:
  virtual std::string serialize() = 0;
};

class IdExpression : public Expression {
 public:
  Identifier id;
  IdExpression() : id() {
  }
  IdExpression(Identifier id) : id(id) {
  }

  std::string serialize() override {
    return id.name();
  }
};

class ConstExpression : public Expression {
 public:
  long double val;
  ConstExpression(long double val) : val(val) {
  }
  std::string serialize() override {
    return fmt::format("{}", val);
  }
};

class ExpressionHandle {
 public:
  std::shared_ptr<Expression> expr;

  ExpressionHandle(int x);

  ExpressionHandle(double x);

  ExpressionHandle(std::shared_ptr<Expression> expr) : expr(expr) {
  }

  ExpressionHandle(Identifier id) {
    expr = std::make_shared<IdExpression>(id);
  }

  Expression *operator->() {
    return expr.get();
  }

  template <typename T>
  Handle<T> cast() const {
    return std::static_pointer_cast<T>(expr);
  }
};

class BinaryOpExpression;

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
};

#define DEFINE_EXPRESSION_OP(op, op_name)                                      \
  Handle<Expression> operator op(ExpressionHandle lhs, ExpressionHandle rhs) { \
    return std::make_shared<BinaryOpExpression>(BinaryType::op_name, lhs,      \
                                                rhs);                          \
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

class StatementList : public Statement {
 public:
  std::vector<Handle<Statement>> statements;
  void insert(const Handle<Statement> &stmt) {
    statements.push_back(stmt);
  }

  DEFINE_ACCEPT
};

class AssignmentStatement : public Statement {
 public:
  Id lhs;
  ExprH rhs;

  AssignmentStatement(Id lhs, ExprH rhs) : lhs(lhs), rhs(rhs) {
  }

  DEFINE_ACCEPT
};

class AllocaStatement : public Statement {
 public:
  Id lhs;
  DataType type;

  AllocaStatement(Id lhs, DataType type) : lhs(lhs), type(type) {
  }
  DEFINE_ACCEPT
};

class BinaryOpStatement : public Statement {
 public:
  BinaryType type;
  Id lhs, rhs1, rhs2;

  BinaryOpStatement(BinaryType type, Id lhs, Id rhs1, Id rhs2)
      : type(type), lhs(lhs), rhs1(rhs1), rhs2(rhs2) {
  }

  DEFINE_ACCEPT
};

class UnaryOpStatement : public Statement {
  DEFINE_ACCEPT
};

class IfStatement : public Statement {
 public:
  ExpressionHandle condition;
  Handle<StatementList> true_statements, false_statements;

  IfStatement(ExpressionHandle condition) : condition(condition) {
  }

  DEFINE_ACCEPT
};

class PrintStatement : public Statement {
 public:
  Id id;

  PrintStatement(Id id) : id(id) {
  }

  DEFINE_ACCEPT
};

class If {
 public:
  Handle<IfStatement> stmt;

  If(ExpressionHandle cond) {
    stmt = std::make_shared<IfStatement>(cond);
    context.builder().insert(stmt);
  }

  If &Then(const std::function<void()> &func) {
    auto _ = context.builder().create_scope(stmt->true_statements);
    func();
    return *this;
  }

  If &Else(const std::function<void()> &func) {
    auto _ = context.builder().create_scope(stmt->false_statements);
    func();
    return *this;
  }
};

class ConstStatement : public Statement {
 public:
  Id id;
  DataType data_type;
  double value;

  ConstStatement(Id id, int32 x) : id(id) {
    data_type = DataType::i32;
    value = x;
  }

  ConstStatement(Id id, float32 x) : id(id) {
    data_type = DataType::f32;
    value = x;
  }

  DEFINE_ACCEPT
};

class ForStatement : public Statement {
 public:
  ExprH begin, end;
  Handle<StatementList> body;
  Id loop_var_id;

  ForStatement(ExprH loop_var, ExprH begin, ExprH end)
      : begin(begin), end(end) {
    loop_var_id = loop_var.cast<IdExpression>()->id;
  }

  DEFINE_ACCEPT
};

class WhileStatement : public Statement {
 public:
  Handle<StatementList> body;

  WhileStatement(const std::function<void()> &cond) {
  }

  DEFINE_ACCEPT
};

void ASTBuilder::insert(const Handle<Statement> &stmt) {
  TC_ASSERT(!stack.empty());
  stack.back()->insert(stmt);
}

void Var(ExpressionHandle &a) {
  current_ast_builder().insert(std::make_shared<AllocaStatement>(
      std::static_pointer_cast<IdExpression>(a.expr)->id, DataType::f32));
}

void Print(const ExpressionHandle &a) {
  context.builder().insert(
      std::make_shared<PrintStatement>(a.cast<IdExpression>()->id));
}

#define DEF_BINARY_OP(Op, name)                                          \
  Identifier operator Op(const Identifier &a, const Identifier &b) {     \
    Identifier c;                                                        \
    current_ast_builder().insert(                                        \
        std::make_shared<BinaryOpStatement>(BinaryType::name, c, a, b)); \
    return c;                                                            \
  }

#undef DEF_BINARY_OP

void Identifier::operator=(const ExpressionHandle &o) {
  context.builder().insert(std::make_shared<AssignmentStatement>(*this, o));
}

class For {
 public:
  For(ExprH i, ExprH s, ExprH e, const std::function<void()> &func) {
    auto stmt = std::make_shared<ForStatement>(i, s, e);
    context.builder().insert(stmt);
    auto _ = context.builder().create_scope(stmt->body);
    func();
  }
};

class While {
 public:
  While(ExprH cond, const std::function<void()> &func) {
    // context.builder().insert()
  }
};

FrontendContext::FrontendContext() {
  root_node = std::make_shared<StatementList>();
  current_builder = std::make_unique<ASTBuilder>(root_node);
}

ExpressionHandle::ExpressionHandle(int x) {
  expr = std::make_shared<ConstExpression>(x);
}

ExpressionHandle::ExpressionHandle(double x) {
  expr = std::make_shared<ConstExpression>(x);
}

TLANG_NAMESPACE_END
