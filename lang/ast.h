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

class Identifier {
 public:
  static int id_counter;

  int id;

  Identifier() {
    id = id_counter++;
  }

  Identifier(int x);

  Identifier(double x);

  void operator=(const Identifier &o);

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

  class Expression {
  public:

  };

  class BinaryOpExpression;

class BinaryOpExpression : public Expression {
public:
  Handle<Expression> lhs, rhs;

  BinaryOpExpression(Handle<Expression> lhs, Handle<Expression> rhs) :lhs(lhs), rhs(rhs){

  }
};

Handle<BinaryOpExpression> operator +(Handle<Expression> lhs, Handle<Expression> rhs) {
return std::make_shared<BinaryOpExpression>(lhs, rhs);
}


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
  Id lhs, rhs;

  AssignmentStatement(Id lhs, Id rhs) : lhs(lhs), rhs(rhs) {
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
  Id condition;
  Handle<StatementList> true_statements, false_statements;

  IfStatement(Id condition) : condition(condition) {
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

  If(Id cond) {
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
  Id loop_var, begin, end;
  Handle<StatementList> body;

  ForStatement(Id loop_var, Id begin, Id end)
      : loop_var(loop_var), begin(begin), end(end) {
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

void Var(Id &a) {
  current_ast_builder().insert(
      std::make_shared<AllocaStatement>(a, DataType::f32));
}

void Print(Id &a) {
  context.builder().insert(std::make_shared<PrintStatement>(a));
}

#define DEF_BINARY_OP(Op, name)                                          \
  Identifier operator Op(const Identifier &a, const Identifier &b) {     \
    Identifier c;                                                        \
    current_ast_builder().insert(                                        \
        std::make_shared<BinaryOpStatement>(BinaryType::name, c, a, b)); \
    return c;                                                            \
  }

DEF_BINARY_OP(+, add);
DEF_BINARY_OP(-, sub);
DEF_BINARY_OP(*, mul);
DEF_BINARY_OP(/, div);
DEF_BINARY_OP(<, cmp_lt);
DEF_BINARY_OP(<=, cmp_le);
DEF_BINARY_OP(>, cmp_gt);
DEF_BINARY_OP(>=, cmp_ge);
DEF_BINARY_OP(==, cmp_eq);
DEF_BINARY_OP(!=, cmp_ne);

#undef DEF_BINARY_OP

void Identifier::operator=(const Identifier &o) {
  context.builder().insert(std::make_shared<AssignmentStatement>(*this, o));
}

class For {
 public:
  For(Id i, Id s, Id e, const std::function<void()> &func) {
    auto stmt = std::make_shared<ForStatement>(i, s, e);
    context.builder().insert(stmt);
    auto _ = context.builder().create_scope(stmt->body);
    func();
  }
};

class While {
 public:
  While(Id cond, const std::function<void()> &func) {
    // context.builder().insert()
  }
};

FrontendContext::FrontendContext() {
  root_node = std::make_shared<StatementList>();
  current_builder = std::make_unique<ASTBuilder>(root_node);
}

Identifier::Identifier(int x) : Identifier() {
  context.builder().insert(std::make_shared<ConstStatement>(*this, x));
}

Identifier::Identifier(double x) : Identifier() {
  context.builder().insert(std::make_shared<ConstStatement>(*this, (float32)x));
}

TLANG_NAMESPACE_END
