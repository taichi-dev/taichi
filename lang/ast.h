#include "util.h"
#include <taichi/util.h>
#include <taichi/testing.h>

TLANG_NAMESPACE_BEGIN

// No Expr nodes - make everything as close to SSA as possible

class ASTBuilder;
class ASTNode;
class Statement;
class StatementList;

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

  ScopeGuard create_scope(const Handle<StatementList> &list) {
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

  Identifier(int x){
      TC_NOT_IMPLEMENTED
      // create const var
  }

  Identifier(double x){
      TC_NOT_IMPLEMENTED
      // create const var
  }

  Identifier
  operator=(const Identifier &o);

  std::string name() {
    return fmt::format("%{}", id);
  }
};

int Identifier::id_counter = 0;

using Id = Identifier;
class AssignmentStatement;
class AllocaStatement;

class ASTVisitor {
 public:
  virtual void visit(StatementList &stmt) = 0;
  virtual void visit(AssignmentStatement &stmt) = 0;
  virtual void visit(AllocaStatement &stmt) = 0;
};

class ASTNode {
 public:
  virtual void accept(ASTVisitor &visitor) {
    TC_NOT_IMPLEMENTED
  }
};

class Statement : public ASTNode {};

class StatementList : public Statement {
 public:
  std::vector<Handle<Statement>> statements;
  void insert(const Handle<Statement> &stmt) {
    statements.push_back(stmt);
  }

  void accept(ASTVisitor &visitor) {
    visitor.visit(*this);
  }
};

class AssignmentStatement : public Statement {
 public:
  Id lhs, rhs;

  AssignmentStatement() {
  }

  void accept(ASTVisitor &visitor) override {
    visitor.visit(*this);
  }
};

class AllocaStatement : public Statement {
 public:
  Id lhs;
  DataType type;

  AllocaStatement(Id lhs, DataType type) : lhs(lhs), type(type) {
  }
  void accept(ASTVisitor &visitor) {
    visitor.visit(*this);
  }
};

class ASTPrinter : public ASTVisitor {
 public:
  static void run(ASTNode &node) {
    auto p = ASTPrinter();
    node.accept(p);
  }

  void visit(StatementList &stmt_list) {
    for (auto &stmt : stmt_list.statements) {
      stmt->accept(*this);
    }
  }

  void visit(AssignmentStatement &assign) {
    fmt::print("{} <- {}", assign.lhs.name(), assign.rhs.name());
  }

  void visit(AllocaStatement &alloca) {
    fmt::print("{} <- alloca {}", alloca.lhs.name(),
               data_type_name(alloca.type));
  }
};

class BinaryOpStmt : public Statement {
 public:
  BinaryType type;
  Id lhs, rhs1, rhs2;

  BinaryOpStmt(BinaryType type, Id lhs, Id rhs1, Id rhs2)
      : type(type), lhs(lhs), rhs1(rhs1), rhs2(rhs2) {
  }

};

class UnaryOp : public ASTNode {};

class IfStatement : public Statement {
  Id condition;
  StatementList true_statements, false_statements;

  IfStatement(Id condition,
              StatementList true_statements,
              StatementList false_statements)
      : condition(condition),
        true_statements(true_statements),
        false_statements(false_statements) {
  }
};

class If {
 public:
  If(Id a) {
  }

  If &Then(const std::function<void()> &func) {
    // create scope...
    func();
    return *this;
  }

  If &Else(const std::function<void()> &func) {
    func();
    return *this;
  }
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
}

#define DEF_BINARY_OP(Op, name)                                      \
  Identifier operator Op(const Identifier &a, const Identifier &b) { \
    Identifier c;                                                    \
    current_ast_builder().insert(                                    \
        std::make_shared<BinaryOpStmt>(BinaryType::name, c, a, b));  \
    return c;                                                        \
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

Identifier Identifier::operator=(const Identifier &o) {
  TC_NOT_IMPLEMENTED;
}

class For {
 public:
  For(Id i, Id s, Id e, const std::function<void()> &func) {
  }
};

class While {
 public:
  While(Id cond, const std::function<void()> &func) {
  }
};

FrontendContext::FrontendContext() {
  root_node = std::make_shared<StatementList>();
  current_builder = std::make_unique<ASTBuilder>(root_node);
}

TLANG_NAMESPACE_END
