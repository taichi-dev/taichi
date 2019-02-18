#include "util.h"
#include <taichi/util.h>
#include <taichi/testing.h>

TLANG_NAMESPACE_BEGIN

// No Expr nodes - make everything as close to SSA as possible

class IRBuilder;
class IRNode;
class Block;
class Statement;
class ConstStatement;
class ForStmt;
class WhileStmt;

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

FrontendContext context;

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

IRBuilder &current_ast_builder() {
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
      return "{" + fmt::format("{}", id) + "}";
    else
      return "{" + name_ + "}";
  }

  bool operator<(const Identifier &o) const {
    return id < o.id;
  }

  bool operator==(const Identifier &o) const {
    return id == o.id;
  }
};

int Identifier::id_counter = 0;

using Ident = Identifier;
class AssignStmt;
class AllocaStmt;
class BinaryOpStmt;
class UnaryOpStmt;
class LocalLoadStmt;
class LocalStoreStmt;
class IfStmt;
class PrintStmt;
class FrontendPrintStmt;
class Block;

using VecStatement = std::vector<std::unique_ptr<Statement>>;

class IRVisitor {
 public:
  bool allow_undefined_visitor;

  IRVisitor() {
    allow_undefined_visitor = false;
  }

#define DEFINE_VISIT(T)          \
  virtual void visit(T *stmt) {  \
    if (allow_undefined_visitor) \
      return;                    \
    else                         \
      TC_NOT_IMPLEMENTED;        \
  }

  DEFINE_VISIT(Block);
  DEFINE_VISIT(AssignStmt);
  DEFINE_VISIT(AllocaStmt);
  DEFINE_VISIT(BinaryOpStmt);
  DEFINE_VISIT(UnaryOpStmt);
  DEFINE_VISIT(LocalLoadStmt);
  DEFINE_VISIT(LocalStoreStmt);
  DEFINE_VISIT(IfStmt);
  DEFINE_VISIT(PrintStmt);
  DEFINE_VISIT(FrontendPrintStmt);
  DEFINE_VISIT(ConstStatement);
  DEFINE_VISIT(ForStmt);
  DEFINE_VISIT(WhileStmt);
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

class Statement : public IRNode {
 public:
  Block *parent;
  DataType type;
  static int id_counter;
  int id;

  Statement() {
    id = id_counter++;
    type = DataType::unknown;
  }

  std::string type_hint() const {
    if (type == DataType::unknown)
      return "";
    else
      return fmt::format("<{}> ", data_type_name(type));
  }

  std::string name() {
    return fmt::format("@{}", id);
  }

  template <typename T>
  bool is() const {
    return dynamic_cast<const T *>(this) != nullptr;
  }
};

int Statement::id_counter = 0;

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

class BinaryOpStmt : public Statement {
 public:
  BinaryType op_type;
  Statement *lhs, *rhs;

  BinaryOpStmt(BinaryType op_type, Statement *lhs, Statement *rhs)
      : op_type(op_type), lhs(lhs), rhs(rhs) {
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
    ret.push_back(
        std::make_unique<BinaryOpStmt>(type, lhs_statement, rhs_statement));
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

class Block : public IRNode {
 public:
  Block *parent;
  std::vector<std::unique_ptr<Statement>> statements;
  std::map<Ident, DataType> local_variables;

  Block() {
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

  DataType lookup_var(Ident ident) const {
    auto ptr = local_variables.find(ident);
    if (ptr != local_variables.end()) {
      return ptr->second;
    } else {
      if (parent) {
        return parent->lookup_var(ident);
      } else {
        return DataType::unknown;
      }
    }
  }

  DEFINE_ACCEPT
};

IRBuilder::ScopeGuard IRBuilder::create_scope(std::unique_ptr<Block> &list) {
  TC_ASSERT(list == nullptr);
  list = std::make_unique<Block>();
  if (!stack.empty()) {
    list->parent = stack.back();
  }
  return ScopeGuard(this, list.get());
}

class AssignStmt : public Statement {
 public:
  ExprH lhs, rhs;
  Ident id;

  AssignStmt(ExprH lhs, ExprH rhs);

  DEFINE_ACCEPT
};

class AllocaStmt : public Statement {
 public:
  Ident ident;

  AllocaStmt(Ident lhs, DataType type) : ident(lhs) {
    this->type = type;
  }
  DEFINE_ACCEPT
};

class UnaryOpStmt : public Statement {
  DEFINE_ACCEPT
};

class LocalLoadStmt : public Statement {
 public:
  Ident ident;

  LocalLoadStmt(Ident ident) : ident(ident) {
  }

  DEFINE_ACCEPT;
};

class LocalStoreStmt : public Statement {
 public:
  Ident ident;
  Statement *stmt;

  LocalStoreStmt(Ident ident, Statement *stmt) : ident(ident), stmt(stmt) {
  }

  DEFINE_ACCEPT;
};

class IfStmt : public Statement {
 public:
  ExpressionHandle condition;
  std::unique_ptr<Block> true_statements, false_statements;

  IfStmt(ExpressionHandle condition) : condition(condition) {
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
  }

  DEFINE_ACCEPT
};

class If {
 public:
  IfStmt *stmt;

  If(ExpressionHandle cond) {
    auto stmt_tmp = std::make_unique<IfStmt>(cond);
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

class ForStmt : public Statement {
 public:
  ExprH begin, end;
  std::unique_ptr<Block> body;
  Ident loop_var_id;

  ForStmt(ExprH loop_var, ExprH begin, ExprH end);

  DEFINE_ACCEPT
};

class WhileStmt : public Statement {
 public:
  std::unique_ptr<Block> body;

  WhileStmt(const std::function<void()> &cond) {
  }

  DEFINE_ACCEPT
};

void IRBuilder::insert(std::unique_ptr<Statement> &&stmt, int location) {
  TC_ASSERT(!stack.empty());
  stack.back()->insert(std::move(stmt), location);
}

void Print(const ExpressionHandle &a) {
  context.builder().insert(std::make_unique<FrontendPrintStmt>(a));
}

#define DEF_BINARY_OP(Op, name)                                      \
  Identifier operator Op(const Identifier &a, const Identifier &b) { \
    Identifier c;                                                    \
    current_ast_builder().insert(                                    \
        std::make_unique<BinaryOpStmt>(BinaryType::name, c, a, b));  \
    return c;                                                        \
  }

#undef DEF_BINARY_OP

void ExprH::operator=(const ExpressionHandle &o) {
  context.builder().insert(std::make_unique<AssignStmt>(*this, o));
}

class For {
 public:
  For(ExprH i, ExprH s, ExprH e, const std::function<void()> &func) {
    auto stmt_unique = std::make_unique<ForStmt>(i, s, e);
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
  root_node = std::make_unique<Block>();
  current_builder = std::make_unique<IRBuilder>(root_node.get());
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

ForStmt::ForStmt(ExprH loop_var, ExprH begin, ExprH end)
    : begin(begin), end(end) {
  loop_var_id = loop_var.cast<IdExpression>()->id;
}

AssignStmt::AssignStmt(ExprH lhs, ExprH rhs) : lhs(lhs), rhs(rhs) {
  id = lhs.cast<IdExpression>()->id;
}

IRNode *FrontendContext::root() {
  return static_cast<IRNode *>(root_node.get());
}

template <typename T>
void Var(ExpressionHandle &a) {
  current_ast_builder().insert(std::make_unique<AllocaStmt>(
      std::static_pointer_cast<IdExpression>(a.expr)->id, get_data_type<T>()));
}

TLANG_NAMESPACE_END
