#include "util.h"
#include <taichi/util.h>
#include <taichi/testing.h>

TLANG_NAMESPACE_BEGIN

// No Expr nodes - make everything as close to SSA as possible

class ASTBuilder;
class ASTNode;

class FrontendContext {
 private:
  ASTBuilder *current_builder;
  ASTNode *root_node;

 public:
  ASTBuilder &builder() {
    return *current_builder;
  }

  ASTNode &root() {
    return *root_node;
  }
};

FrontendContext context;

class Statement;
class StatementList;

class ASTBuilder {
 private:
  std::vector<Handle<StatementList>> stack;
  Handle<StatementList> stmt_list;

 public:
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

ASTBuilder *ast_builder = nullptr;

ASTBuilder &current_ast_builder() {
  TC_ASSERT(ast_builder != nullptr);
  return *ast_builder;
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
};

int Identifier::id_counter = 0;

using Id = Identifier;

class ASTVisitor;

class ASTNode {
 public:
  virtual void accept(ASTVisitor &visitor) {
    TC_NOT_IMPLEMENTED
  }
  virtual void print() {
    TC_NOT_IMPLEMENTED
  }
};

class Statement : public ASTNode {};

class StatementList : public Statement {
  std::vector<Handle<Statement>> statements;

 public:
  void insert(const Handle<Statement> &stmt) {
    statements.push_back(stmt);
  }
};

class AssignmentStatement : public Statement {
 public:
  AssignmentStatement() {
  }
};

class ASTVisitor {
 public:
  virtual void visit(ASTNode &node) = 0;
};

class ASTPrinter : public ASTVisitor {
 public:
  void visit(ASTNode &node) override {
    node.print();
    node.accept(*this);
  }

  static void run(ASTNode &node) {
    auto p = ASTPrinter();
    node.accept(p);
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
  stmt_list->insert(stmt);
}

void Var(Id &a) {
  current_ast_builder();
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

void print_ast() {
  auto visit = [](ASTNode *node) {

  };
}

ASTNode &get_ast_node() {
}

auto test_ast = []() {
  Id a, b, i, j;

  Var(a);
  Var(b);
  Var(i);
  Var(j);

  a = a + 1;

  If(a > 5)
      .Then([&] {
        b = (b + 1) / 3;
        b = b * 3;
      })
      .Else([&] {
        b = b + 2;
        b = b - 4;
      });

  For(i, 0, 100, [&] {
    For(j, 0, 200, [&] {
      Id k = i + j;
      While(k < 500, [&] { Print(k); });
    });
  });

  Print(b);

  ASTPrinter::run(context.root());
};

TC_REGISTER_TASK(test_ast);

TLANG_NAMESPACE_END
