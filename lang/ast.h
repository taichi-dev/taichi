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
  std::unique_ptr<StatementList> root_node;

 public:
  FrontendContext();

  ASTBuilder &builder() {
    return *current_builder;
  }

  ASTNode *root();
};

FrontendContext context;

class ASTBuilder {
 private:
  std::vector<StatementList *> stack;

 public:
  ASTBuilder(StatementList *initial) {
    stack.push_back(initial);
  }

  void insert(std::unique_ptr<Statement> &&stmt, int location = -1);

  struct ScopeGuard {
    ASTBuilder *builder;
    StatementList *list;
    ScopeGuard(ASTBuilder *builder, StatementList *list)
        : builder(builder), list(list) {
      builder->stack.push_back(list);
    }

    ~ScopeGuard() {
      builder->stack.pop_back();
    }
  };

  ScopeGuard create_scope(std::unique_ptr<StatementList> &list) {
    TC_ASSERT(list == nullptr);
    list = std::make_unique<StatementList>();
    return ScopeGuard(this, list.get());
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
  std::string name_;

  int id;

  Identifier(std::string name_ = "") : name_(name_) {
    id = id_counter++;
  }

  std::string name() {
    if (name_.empty())
      return fmt::format("%{}", id);
    else
      return "%" + name_;
  }
};

int Identifier::id_counter = 0;

using Id = Identifier;
class AssignmentStatement;
class AllocaStatement;
class BinaryOpStatement;
class UnaryOpStatement;
class LocalLoadStmt;
class LocalStoreStmt;
class IfStatement;
class PrintStatement;
class StatementList;

using VecStatement = std::vector<std::unique_ptr<Statement>>;

class ASTVisitor {
 public:
#define DEFINE_VISIT(T)         \
  virtual void visit(T *stmt) { \
    TC_NOT_IMPLEMENTED;         \
  }

  DEFINE_VISIT(StatementList);
  DEFINE_VISIT(AssignmentStatement);
  DEFINE_VISIT(AllocaStatement);
  DEFINE_VISIT(BinaryOpStatement);
  DEFINE_VISIT(UnaryOpStatement);
  DEFINE_VISIT(LocalLoadStmt);
  DEFINE_VISIT(LocalStoreStmt);
  DEFINE_VISIT(IfStatement);
  DEFINE_VISIT(PrintStatement);
  DEFINE_VISIT(ConstStatement);
  DEFINE_VISIT(ForStatement);
  DEFINE_VISIT(WhileStatement);
};

class ASTNode {
 public:
  virtual void accept(ASTVisitor *visitor) {
    TC_NOT_IMPLEMENTED
  }
};

#define DEFINE_ACCEPT                \
  void accept(ASTVisitor *visitor) { \
    visitor->visit(this);            \
  }

class Statement : public ASTNode {
 public:
  StatementList *parent;
  DataType type;
  Id ret;

  std::string name() {
    return ret.name();
  }
};

// always a tree - used as rvalues
class Expression {
 public:
  virtual std::string serialize() = 0;
  virtual void flatten(VecStatement &ret) {
    TC_NOT_IMPLEMENTED;
  };
};

class LocalLoadStmt;

class ExpressionHandle {
 public:
  std::shared_ptr<Expression> expr;

  ExpressionHandle(int x);

  ExpressionHandle(double x);

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

  void operator=(const ExpressionHandle &o);

  std::string serialize() const {
    return expr->serialize();
  }
};

class BinaryOpStatement : public Statement {
 public:
  BinaryType type;
  Statement *lhs, *rhs;

  BinaryOpStatement(BinaryType type, Statement *lhs, Statement *rhs)
      : type(type), lhs(lhs), rhs(rhs) {
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
    lhs->flatten(ret);
    auto lhs_statement = ret.back().get();
    rhs->flatten(ret);
    auto rhs_statement = ret.back().get();
    ret.push_back(std::make_unique<BinaryOpStatement>(type, lhs_statement,
                                                      rhs_statement));
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
  std::vector<std::unique_ptr<Statement>> statements;

  void insert(std::unique_ptr<Statement> &&stmt, int location = -1) {
    stmt->parent = this;
    if (location == -1) {
      statements.push_back(std::move(stmt));
    } else {
      statements.insert(statements.begin() + location, std::move(stmt));
    }
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

  DEFINE_ACCEPT
};

class AssignmentStatement : public Statement {
 public:
  ExprH lhs, rhs;
  Id id;

  AssignmentStatement(ExprH lhs, ExprH rhs);

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

class UnaryOpStatement : public Statement {
  DEFINE_ACCEPT
};

class LocalLoadStmt : public Statement {
 public:
  Id id;

  LocalLoadStmt(Id id) : id(id) {
  }

  DEFINE_ACCEPT;
};

class LocalStoreStmt : public Statement {
 public:
  Id id;
  Statement *stmt;

  LocalStoreStmt(Id id, Statement *stmt) : id(id), stmt(stmt) {
  }

  DEFINE_ACCEPT;
};

class IfStatement : public Statement {
 public:
  ExpressionHandle condition;
  std::unique_ptr<StatementList> true_statements, false_statements;

  IfStatement(ExpressionHandle condition) : condition(condition) {
  }

  DEFINE_ACCEPT
};

class PrintStatement : public Statement {
 public:
  ExprH expr;

  PrintStatement(ExprH expr) : expr(expr) {
  }

  DEFINE_ACCEPT
};

class If {
 public:
  IfStatement *stmt;

  If(ExpressionHandle cond) {
    auto stmt_tmp = std::make_unique<IfStatement>(cond);
    stmt = stmt_tmp.get();
    context.builder().insert(std::move(stmt_tmp));
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
  DataType data_type;
  double value;

  ConstStatement(int32 x) {
    data_type = DataType::i32;
    value = x;
  }

  ConstStatement(float32 x) {
    data_type = DataType::f32;
    value = x;
  }

  DEFINE_ACCEPT
};

class ForStatement : public Statement {
 public:
  ExprH begin, end;
  std::unique_ptr<StatementList> body;
  Id loop_var_id;

  ForStatement(ExprH loop_var, ExprH begin, ExprH end);

  DEFINE_ACCEPT
};

class WhileStatement : public Statement {
 public:
  std::unique_ptr<StatementList> body;

  WhileStatement(const std::function<void()> &cond) {
  }

  DEFINE_ACCEPT
};

void ASTBuilder::insert(std::unique_ptr<Statement> &&stmt, int location) {
  TC_ASSERT(!stack.empty());
  stack.back()->insert(std::move(stmt), location);
}

void Print(const ExpressionHandle &a) {
  context.builder().insert(std::make_unique<PrintStatement>(a));
}

#define DEF_BINARY_OP(Op, name)                                          \
  Identifier operator Op(const Identifier &a, const Identifier &b) {     \
    Identifier c;                                                        \
    current_ast_builder().insert(                                        \
        std::make_unique<BinaryOpStatement>(BinaryType::name, c, a, b)); \
    return c;                                                            \
  }

#undef DEF_BINARY_OP

void ExprH::operator=(const ExpressionHandle &o) {
  context.builder().insert(std::make_unique<AssignmentStatement>(*this, o));
}

class For {
 public:
  For(ExprH i, ExprH s, ExprH e, const std::function<void()> &func) {
    auto stmt_unique = std::make_unique<ForStatement>(i, s, e);
    auto stmt = stmt_unique.get();
    context.builder().insert(std::move(stmt_unique));
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
  root_node = std::make_unique<StatementList>();
  current_builder = std::make_unique<ASTBuilder>(root_node.get());
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
    ret.push_back(std::make_unique<LocalLoadStmt>(id));
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

  void flatten(VecStatement &ret) override {
    ret.push_back(std::make_unique<ConstStatement>((float32)val));
  }
};

ExpressionHandle::ExpressionHandle(int x) {
  expr = std::make_shared<ConstExpression>(x);
}

ExpressionHandle::ExpressionHandle(double x) {
  expr = std::make_shared<ConstExpression>(x);
}

ExpressionHandle::ExpressionHandle(Identifier id) {
  expr = std::make_shared<IdExpression>(id);
}

ForStatement::ForStatement(ExprH loop_var, ExprH begin, ExprH end)
    : begin(begin), end(end) {
  loop_var_id = loop_var.cast<IdExpression>()->id;
}

AssignmentStatement::AssignmentStatement(ExprH lhs, ExprH rhs)
    : lhs(lhs), rhs(rhs) {
  id = lhs.cast<IdExpression>()->id;
}

ASTNode *FrontendContext::root() {
  return static_cast<ASTNode *>(root_node.get());
}

void Var(ExpressionHandle &a) {
  current_ast_builder().insert(std::make_unique<AllocaStatement>(
      std::static_pointer_cast<IdExpression>(a.expr)->id, DataType::f32));
}

TLANG_NAMESPACE_END
