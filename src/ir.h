#pragma once

#include <atomic>
#include <taichi/util.h>
#include "util.h"
#include "snode.h"

TLANG_NAMESPACE_BEGIN

class DiffRange {
 public:
  bool related;
  int low, high;

  DiffRange() : DiffRange(false) {
  }

  DiffRange(bool related) : DiffRange(related, 0, 0) {
    TC_ASSERT(related == false);
  }

  DiffRange(bool related, int low) : DiffRange(related, low, low + 1) {
  }

  DiffRange(bool related, int low, int high)
      : related(related), low(low), high(high) {
    if (!related) {
      this->low = this->high = 0;
    }
  }

  bool certain() {
    TC_ASSERT(related);
    return high == low + 1;
  }
};

class IRBuilder;
class IRNode;
class Block;
class Stmt;
using pStmt = std::unique_ptr<Stmt>;
class DiffRange;

class SNode;
using ScratchPadOptions = std::vector<std::pair<int, SNode *>>;
class Expression;
class Expr;
class ExpressionGroup;

// Frontend statements
class FrontendIfStmt;
class FrontendForStmt;
class FrontendPrintStmt;
class FrontendWhileStmt;
class FrontendAllocaStmt;
class FrontendAssignStmt;
class FrontendAtomicStmt;
class FrontendEvalStmt;
class FrontendSNodeOpStmt;  // activate, deactivate, append, clear
class FrontendAssertStmt;

// Midend statement

// Without per-lane attributes
class RangeForStmt;
class StructForStmt;
class IfStmt;
class WhileStmt;
class WhileControlStmt;

class ConstStmt;
class AllocaStmt;
class UnaryOpStmt;
class BinaryOpStmt;
class TrinaryOpStmt;
class PrintStmt;
class RandStmt;
class GlobalLoadStmt;
class GlobalStoreStmt;
class AtomicOpStmt;
class LocalStoreStmt;
class SNodeOpStmt;
class RangeAssumptionStmt;
class AssertStmt;

// With per-lane attributes
class LocalLoadStmt;
class GlobalPtrStmt;
class ElementShuffleStmt;

// Pragma statements
class PragmaSLPStmt;
class ScratchPads;

// IR passes
namespace irpass {

void re_id(IRNode *root);
void eliminate_dup(IRNode *root);
void print(IRNode *root);
void lower(IRNode *root);
void typecheck(IRNode *root);
void loop_vectorize(IRNode *root);
void slp_vectorize(IRNode *root);
void vector_split(IRNode *root, int max_width, bool serial_schedule);
void replace_all_usages_with(IRNode *root, Stmt *old_stmt, Stmt *new_stmt);
std::unique_ptr<ScratchPads> initialize_scratch_pad(StructForStmt *root);

}  // namespace irpass

// Analysis
namespace analysis {
DiffRange value_diff(Stmt *stmt, int lane, Stmt *alloca);
}

IRBuilder &current_ast_builder();

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

  std::string short_str() const {
    return fmt::format("{}x{}", data_type_short_name(data_type), width);
  }

  std::string str() const {
    return fmt::format("{}x{}", data_type_name(data_type), width);
  }
};

class DecoratorRecorder {
 public:
  int vectorize;
  int parallelize;
  ScratchPadOptions scratch_opt;
  int block_size;
  bool uniform;

  DecoratorRecorder() {
    reset();
  }

  void reset() {
    vectorize = -1;
    parallelize = 0;
    uniform = false;
    scratch_opt.clear();
    block_size = 0;
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

  std::unique_ptr<Block> get_root() {
    return std::move(root_node);
  }
};

extern std::unique_ptr<FrontendContext> context;

class IRBuilder {
 private:
  std::vector<Block *> stack;

 public:
  IRBuilder(Block *initial) {
    stack.push_back(initial);
  }

  void insert(std::unique_ptr<Stmt> &&stmt, int location = -1);

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
};

IRBuilder &current_ast_uilder();

inline Expr load_if_ptr(const Expr &ptr);

class Identifier {
 public:
  static int id_counter;
  std::string name_;

  int id;

  // Multiple identifiers can share the same name but must have different id's
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

  VecStatement(VecStatement &&o) {
    stmts = std::move(o.stmts);
  }

  Stmt *push_back(pStmt &&stmt) {
    auto ret = stmt.get();
    stmts.push_back(std::move(stmt));
    return ret;
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
  virtual void visit(Stmt *stmt) {
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
  DEFINE_VISIT(FrontendAllocaStmt);
  DEFINE_VISIT(FrontendPrintStmt);
  DEFINE_VISIT(FrontendForStmt);
  DEFINE_VISIT(FrontendWhileStmt);
  DEFINE_VISIT(FrontendAssignStmt);
  DEFINE_VISIT(FrontendAtomicStmt);
  DEFINE_VISIT(FrontendSNodeOpStmt);
  DEFINE_VISIT(FrontendEvalStmt);
  DEFINE_VISIT(FrontendAssertStmt);

  DEFINE_VISIT(SNodeOpStmt);
  DEFINE_VISIT(AllocaStmt);
  DEFINE_VISIT(UnaryOpStmt);
  DEFINE_VISIT(LocalLoadStmt);
  DEFINE_VISIT(BinaryOpStmt);
  DEFINE_VISIT(TrinaryOpStmt);
  DEFINE_VISIT(AtomicOpStmt);
  DEFINE_VISIT(LocalStoreStmt);
  DEFINE_VISIT(GlobalLoadStmt);
  DEFINE_VISIT(GlobalStoreStmt);
  DEFINE_VISIT(GlobalPtrStmt);
  DEFINE_VISIT(IfStmt);
  DEFINE_VISIT(PrintStmt);
  DEFINE_VISIT(ConstStmt);
  DEFINE_VISIT(RangeForStmt);
  DEFINE_VISIT(StructForStmt);
  DEFINE_VISIT(WhileStmt);
  DEFINE_VISIT(WhileControlStmt);
  DEFINE_VISIT(RandStmt);
  DEFINE_VISIT(RangeAssumptionStmt);
  DEFINE_VISIT(AssertStmt);

  DEFINE_VISIT(PragmaSLPStmt);
  DEFINE_VISIT(ElementShuffleStmt);
};

class IRNode {
 public:
  virtual void accept(IRVisitor *visitor) {
    TC_NOT_IMPLEMENTED
  }
  virtual ~IRNode() {
  }
};

#define DEFINE_ACCEPT                        \
  void accept(IRVisitor *visitor) override { \
    visitor->visit(this);                    \
  }

template <typename T>
struct LaneAttribute {
  std::vector<T> data;

  LaneAttribute() {
  }

  LaneAttribute(const std::vector<T> &data) : data(data) {
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

  LaneAttribute slice(int begin, int end) {
    return LaneAttribute(
        std::vector<T>(data.begin() + begin, data.begin() + end));
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
    } else if (bracket == "[") {
      ret += "]";
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

class Stmt : public IRNode {
 public:
  static std::atomic<int> instance_id_counter;
  int instance_id;
  int id;
  Block *parent;
  std::vector<Stmt **> operands;

  int &width() {
    return ret_type.width;
  }

  virtual bool is_container_statement() const {
    return false;
  }

  DataType &element_type() {
    return ret_type.data_type;
  }

  VectorType ret_type;

  Stmt(const Stmt &stmt) = delete;

  Stmt() {
    parent = nullptr;
    instance_id = instance_id_counter++;
    id = instance_id;
  }

  std::string ret_data_type_name() const {
    return ret_type.str();
  }

  std::string type_hint() const {
    if (ret_type.data_type == DataType::unknown)
      return "";
    else
      return fmt::format("<{}> ", ret_type.short_str());
  }

  std::string name() const {
    return fmt::format("${}", id);
  }

  std::string raw_name() const {
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

  Stmt *&operand(int i) {
    TC_ASSERT(0 <= i && i < (int)operands.size());
    return *operands[i];
  }

  void add_operand(Stmt *&stmt) {
    operands.push_back(&stmt);
  }

  IRNode *get_ir_root();

  virtual void repeat(int factor) {
    ret_type.width *= factor;
  }

  void replace_with(Stmt *new_stmt);

  virtual void replace_operand_with(Stmt *old_stmt, Stmt *new_stmt);

  void insert_before_me(std::unique_ptr<Stmt> &&new_stmt);

  void insert_after_me(std::unique_ptr<Stmt> &&new_stmt);

  virtual bool integral_operands() const {
    return true;
  }

  template <typename T, typename... Args>
  static pStmt make(Args &&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
  }

  virtual ~Stmt() {
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

  virtual bool is_lvalue() const {
    return false;
  }

  virtual ~Expression() {
  }
};

class Expr {
 public:
  std::shared_ptr<Expression> expr;
  bool const_value;
  bool atomic;

  Expr() {
    const_value = false;
    atomic = false;
  }

  Expr(int32 x);

  Expr(float32 x);

  Expr(std::shared_ptr<Expression> expr) : Expr() {
    this->expr = expr;
  }

  Expr(const Expr &o) : Expr() {
    set(o);
    const_value = o.const_value;
  }

  Expr(Expr &&o) : Expr() {
    set(o);
    const_value = o.const_value;
    atomic = o.atomic;
  }

  Expr(Identifier id);

  void set(const Expr &o) {
    expr = o.expr;
  }

  Expression *operator->() {
    return expr.get();
  }

  template <typename T>
  Handle<T> cast() const {
    TC_ASSERT(expr != nullptr);
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
  Expr operator!();

  Expr eval() const;

  template <typename T, typename... Args>
  static Expr make(Args &&... args) {
    return Expr(std::make_shared<T>(std::forward<Args>(args)...));
  }

  Expr parent() const;

  SNode *snode() const;
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

  ExpressionGroup(const Expr &a, ExpressionGroup b) {
    exprs = b.exprs;
    exprs.insert(exprs.begin(), a);
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

  std::string serialize() {
    std::string ret;
    for (int i = 0; i < (int)exprs.size(); i++) {
      ret += exprs[i].serialize();
      if (i + 1 < (int)exprs.size()) {
        ret += ", ";
      }
    }
    return ret;
  }

  ExpressionGroup loaded() const;
};

inline ExpressionGroup operator,(const Expr &a, const Expr &b) {
  return ExpressionGroup(a, b);
}

inline ExpressionGroup operator,(const ExpressionGroup &a, const Expr &b) {
  return ExpressionGroup(a, b);
}

class FrontendAllocaStmt : public Stmt {
 public:
  Ident ident;

  FrontendAllocaStmt(Ident lhs, DataType type) : ident(lhs) {
    ret_type = VectorType(1, type);
  }

  DEFINE_ACCEPT
};

class AllocaStmt : public Stmt {
 public:
  AllocaStmt(DataType type) {
    ret_type = VectorType(1, type);
  }

  AllocaStmt(int width, DataType type) {
    ret_type = VectorType(width, type);
  }

  DEFINE_ACCEPT
};

// updates mask, break if no active
class WhileControlStmt : public Stmt {
 public:
  Stmt *mask;
  Stmt *cond;
  WhileControlStmt(Stmt *mask, Stmt *cond) : mask(mask), cond(cond) {
  }
  DEFINE_ACCEPT;
};

class UnaryOpStmt : public Stmt {
 public:
  UnaryType op_type;
  Stmt *rhs;
  DataType cast_type;
  bool cast_by_value = true;

  UnaryOpStmt(UnaryType op_type, Stmt *rhs) : op_type(op_type), rhs(rhs) {
    add_operand(this->rhs);
    cast_type = DataType::unknown;
    cast_by_value = true;
  }

  bool same_operation(UnaryOpStmt *o) const {
    if (op_type == o->op_type) {
      if (op_type == UnaryType::cast) {
        return cast_type == o->cast_type;
      } else {
        return true;
      }
    }
    return false;
  }

  DEFINE_ACCEPT
};

class RandStmt : public Stmt {
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
  bool cast_by_value;

  UnaryOpExpression(UnaryType type, Expr rhs)
      : type(type), rhs(load_if_ptr(rhs)) {
    cast_type = DataType::unknown;
    cast_by_value = true;
  }

  std::string serialize() override {
    if (type == UnaryType::cast) {
      std::string reint = cast_by_value ? "" : "reinterpret_";
      return fmt::format("({}{}<{}> {})", reint, unary_type_name(type),
                         data_type_name(cast_type), rhs->serialize());
    } else {
      return fmt::format("({} {})", unary_type_name(type), rhs->serialize());
    }
  }

  void flatten(VecStatement &ret) override {
    rhs->flatten(ret);
    auto unary = std::make_unique<UnaryOpStmt>(type, rhs->stmt);
    if (type == UnaryType::cast) {
      unary->cast_type = cast_type;
      unary->cast_by_value = cast_by_value;
    }
    stmt = unary.get();
    ret.push_back(std::move(unary));
  }
};

class BinaryOpStmt : public Stmt {
 public:
  BinaryType op_type;
  Stmt *lhs, *rhs;

  BinaryOpStmt(BinaryType op_type, Stmt *lhs, Stmt *rhs)
      : op_type(op_type), lhs(lhs), rhs(rhs) {
    add_operand(this->lhs);
    add_operand(this->rhs);
  }

  DEFINE_ACCEPT
};

class TrinaryOpStmt : public Stmt {
 public:
  TrinaryType op_type;
  Stmt *op1, *op2, *op3;

  TrinaryOpStmt(TrinaryType op_type, Stmt *op1, Stmt *op2, Stmt *op3)
      : op_type(op_type), op1(op1), op2(op2), op3(op3) {
    add_operand(this->op1);
    add_operand(this->op2);
    add_operand(this->op3);
  }

  DEFINE_ACCEPT
};

class AtomicOpStmt : public Stmt {
 public:
  AtomicType op_type;
  Stmt *dest, *val;

  AtomicOpStmt(AtomicType op_type, Stmt *dest, Stmt *val)
      : op_type(op_type), dest(dest), val(val) {
    add_operand(this->dest);
    add_operand(this->val);
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

class TrinaryOpExpression : public Expression {
 public:
  TrinaryType type;
  Expr op1, op2, op3;

  TrinaryOpExpression(TrinaryType type,
                      const Expr &op1,
                      const Expr &op2,
                      const Expr &op3)
      : type(type) {
    this->op1.set(load_if_ptr(op1));
    this->op2.set(load_if_ptr(op2));
    this->op3.set(load_if_ptr(op3));
  }

  std::string serialize() override {
    return fmt::format("{}({} {} {})", trinary_type_name(type),
                       op1->serialize(), op2->serialize(), op3->serialize());
  }

  void flatten(VecStatement &ret) override {
    // if (stmt)
    //  return;
    op1->flatten(ret);
    op2->flatten(ret);
    op3->flatten(ret);
    ret.push_back(
        std::make_unique<TrinaryOpStmt>(type, op1->stmt, op2->stmt, op3->stmt));
    stmt = ret.back().get();
  }
};

class GlobalPtrStmt : public Stmt {
 public:
  LaneAttribute<SNode *> snodes;
  std::vector<Stmt *> indices;

  GlobalPtrStmt(const LaneAttribute<SNode *> &snodes,
                const std::vector<Stmt *> &indices)
      : snodes(snodes), indices(indices) {
    for (int i = 0; i < (int)snodes.size(); i++) {
      TC_ASSERT(snodes[i] != nullptr);
      TC_ASSERT(snodes[0]->dt == snodes[i]->dt);
    }
    for (int i = 0; i < (int)indices.size(); i++) {
      add_operand(this->indices[i]);
    }
    width() = snodes.size();
    element_type() = snodes[0]->dt;
  }

  DEFINE_ACCEPT
};

class GlobalVariableExpression : public Expression {
 public:
  Identifier ident;
  DataType dt;
  SNode *snode;
  bool has_ambient;
  TypedConstant ambient_value;

  GlobalVariableExpression(DataType dt, Ident ident) : ident(ident), dt(dt) {
    snode = nullptr;
    has_ambient = false;
  }

  GlobalVariableExpression(SNode *snode) : snode(snode) {
    dt = DataType::unknown;
    snode = nullptr;
    has_ambient = false;
  }

  std::string serialize() override {
    return "#" + ident.name();
  }

  void flatten(VecStatement &ret) override {
    TC_ASSERT(snode->num_active_indices == 0);
    auto ptr = Stmt::make<GlobalPtrStmt>(LaneAttribute<SNode *>(snode),
                                         std::vector<Stmt *>());
    ret.push_back(std::move(ptr));
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

  bool is_lvalue() const override {
    return true;
  }
};

#include "expression.h"

Expr select(const Expr &cond, const Expr &true_val, const Expr &false_val);

Expr operator-(Expr expr);

Expr operator~(Expr expr);

Expr exp(const Expr &expr);

Expr log(const Expr &expr);

// Value cast
template <typename T>
Expr cast(Expr input);

template <typename T>
Expr bit_cast(Expr input);

class Block : public IRNode {
 public:
  Block *parent;
  std::vector<std::unique_ptr<Stmt>> statements, trash_bin;
  std::map<Ident, Stmt *> local_var_alloca;
  Stmt *mask_var;

  Block() {
    mask_var = nullptr;
    parent = nullptr;
  }

  bool has_container_statements() {
    for (auto &s : statements) {
      if (s->is_container_statement())
        return true;
    }
    return false;
  }

  int locate(Stmt *stmt) {
    for (int i = 0; i < (int)statements.size(); i++) {
      if (statements[i].get() == stmt) {
        return i;
      }
    }
    TC_ERROR("Stmt not found");
    return -1;
  }

  void erase(int location);

  void insert(std::unique_ptr<Stmt> &&stmt, int location = -1);

  void replace_statements_in_range(int start, int end, VecStatement &&stmts);

  void set_statements(VecStatement &&stmts) {
    statements.clear();
    for (int i = 0; i < (int)stmts.size(); i++) {
      insert(std::move(stmts[i]), i);
    }
  }

  void replace_with(Stmt *old_statement, std::unique_ptr<Stmt> &&new_statement);

  void replace_with(Stmt *old_statement, VecStatement &new_statements) {
    int location = -1;
    for (int i = 0; i < (int)statements.size(); i++) {
      if (old_statement == statements[i].get()) {
        location = i;
        break;
      }
    }
    TC_ASSERT(location != -1);
    old_statement->replace_with(new_statements.back().get());
    statements.erase(statements.begin() + location);
    for (int i = (int)new_statements.size() - 1; i >= 0; i--) {
      insert(std::move(new_statements[i]), location);
    }
  }

  Stmt *lookup_var(Ident ident) const;

  Stmt *mask();

  DEFINE_ACCEPT
};

class FrontendAtomicStmt : public Stmt {
 public:
  AtomicType op_type;
  Expr dest, val;

  FrontendAtomicStmt(AtomicType op_type, Expr dest, Expr val);

  DEFINE_ACCEPT
};

class FrontendSNodeOpStmt : public Stmt {
 public:
  SNodeOpType op_type;
  SNode *snode;
  ExpressionGroup indices;
  Expr val;

  FrontendSNodeOpStmt(SNodeOpType op_type,
                      SNode *snode,
                      ExpressionGroup indices,
                      Expr val = Expr(nullptr))
      : op_type(op_type), snode(snode), indices(indices.loaded()), val(val) {
    if (val.expr != nullptr) {
      TC_ASSERT(op_type == SNodeOpType::append);
      this->val.set(load_if_ptr(val));
    } else {
      TC_ASSERT(op_type != SNodeOpType::append);
    }
  }

  DEFINE_ACCEPT
};

class SNodeOpStmt : public Stmt {
 public:
  SNodeOpType op_type;
  LaneAttribute<SNode *> snodes;
  std::vector<Stmt *> indices;
  Stmt *val;

  SNodeOpStmt(SNodeOpType op_type,
              const LaneAttribute<SNode *> &snodes,
              const std::vector<Stmt *> &indices,
              Stmt *val = nullptr)
      : op_type(op_type), snodes(snodes), indices(indices), val(val) {
    TC_ASSERT_INFO(snodes.size() == 1, "SNodeOpStmt cannot be vectorized");
    TC_ASSERT((val == nullptr) != (op_type == SNodeOpType::append));
    for (int i = 0; i < (int)snodes.size(); i++) {
      TC_ASSERT(snodes[i] != nullptr);
      TC_ASSERT(snodes[0]->dt == snodes[i]->dt);
    }
    for (int i = 0; i < (int)indices.size(); i++) {
      add_operand(this->indices[i]);
    }
    if (val) {
      add_operand(this->val);
    }
    width() = snodes.size();
    element_type() = snodes[0]->dt;
  }

  DEFINE_ACCEPT
};

class FrontendAssertStmt : public Stmt {
 public:
  std::string text;
  Expr val;

  FrontendAssertStmt(const std::string &text, Expr val) : text(text), val(val) {
  }

  DEFINE_ACCEPT
};

class AssertStmt : public Stmt {
 public:
  std::string text;
  Stmt *val;

  AssertStmt(const std::string &text, Stmt *val) : text(text), val(val) {
    TC_ASSERT(val);
  }

  DEFINE_ACCEPT
};
class RangeAssumptionStmt : public Stmt {
 public:
  Stmt *input;
  Stmt *base;
  int low, high;

  RangeAssumptionStmt(Stmt *input, Stmt *base, int low, int high)
      : input(input), base(base), low(low), high(high) {
    add_operand(this->input);
    add_operand(this->base);
  }

  DEFINE_ACCEPT
};

class FrontendAssignStmt : public Stmt {
 public:
  Expr lhs, rhs;

  FrontendAssignStmt(Expr lhs, Expr rhs);

  DEFINE_ACCEPT
};

class GlobalLoadStmt : public Stmt {
 public:
  Stmt *ptr;

  GlobalLoadStmt(Stmt *ptr) : ptr(ptr) {
    add_operand(this->ptr);
  }

  DEFINE_ACCEPT;
};

class GlobalStoreStmt : public Stmt {
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

class LocalLoadStmt : public Stmt {
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

  Stmt *previous_store_or_alloca_in_block();

  DEFINE_ACCEPT;
};

class LocalStoreStmt : public Stmt {
 public:
  Stmt *ptr;
  Stmt *data;

  // LaneAttribute<Stmt *> data;

  LocalStoreStmt(Stmt *ptr, Stmt *data) : ptr(ptr), data(data) {
    add_operand(this->ptr);
    add_operand(this->data);
  }

  DEFINE_ACCEPT;
};

class IfStmt : public Stmt {
 public:
  Stmt *cond;
  Stmt *true_mask, *false_mask;
  std::unique_ptr<Block> true_statements, false_statements;

  IfStmt(Stmt *cond) : cond(cond) {
    add_operand(this->cond);
  }

  bool is_container_statement() const override {
    return true;
  }

  DEFINE_ACCEPT
};

class FrontendIfStmt : public Stmt {
 public:
  Expr condition;
  std::unique_ptr<Block> true_statements, false_statements;

  FrontendIfStmt(Expr condition) : condition(condition) {
  }

  bool is_container_statement() const override {
    return true;
  }

  DEFINE_ACCEPT
};

class FrontendPrintStmt : public Stmt {
 public:
  Expr expr;
  std::string str;

  FrontendPrintStmt(Expr expr, std::string str)
      : expr(load_if_ptr(expr)), str(str) {
  }

  DEFINE_ACCEPT
};

class FrontendEvalStmt : public Stmt {
 public:
  Expr expr;
  Expr eval_expr;

  FrontendEvalStmt(Expr expr) : expr(load_if_ptr(expr)) {
  }

  DEFINE_ACCEPT
};

class PrintStmt : public Stmt {
 public:
  Stmt *stmt;
  std::string str;

  PrintStmt(Stmt *stmt, std::string str) : stmt(stmt), str(str) {
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

class ConstStmt : public Stmt {
 public:
  LaneAttribute<TypedConstant> val;

  ConstStmt(const LaneAttribute<TypedConstant> &val) : val(val) {
    width() = val.size();
    element_type() = val[0].dt;
    for (int i = 0; i < ret_type.width; i++) {
      TC_ASSERT(val[0].dt == val[i].dt);
    }
  }

  void repeat(int factor) override {
    Stmt::repeat(factor);
    val.repeat(factor);
  }

  DEFINE_ACCEPT
};

class FrontendForStmt : public Stmt {
 public:
  Expr begin, end;
  Expr global_var;
  std::unique_ptr<Block> body;
  std::vector<Ident> loop_var_id;
  int vectorize;
  int parallelize;
  ScratchPadOptions scratch_opt;
  int block_size;

  bool is_ranged() const {
    if (global_var.expr == nullptr) {
      return true;
    } else {
      return false;
    }
  }

  FrontendForStmt(ExpressionGroup loop_var, Expr global_var);

  FrontendForStmt(ExpressionGroup loop_var, Expr begin, Expr end);

  bool is_container_statement() const override {
    return true;
  }

  DEFINE_ACCEPT
};

// General range for
class RangeForStmt : public Stmt {
 public:
  Stmt *loop_var;
  Stmt *begin, *end;
  std::unique_ptr<Block> body;
  int vectorize;
  int parallelize;
  int block_size;

  RangeForStmt(Stmt *loop_var,
               Stmt *begin,
               Stmt *end,
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
    block_size = 256;
  }

  bool is_container_statement() const override {
    return true;
  }

  DEFINE_ACCEPT
};

// for stmt over a structural node
class StructForStmt : public Stmt {
 public:
  std::vector<Stmt *> loop_vars;
  SNode *snode;
  std::unique_ptr<Block> body;
  int vectorize;
  int parallelize;
  int block_size;
  ScratchPadOptions scratch_opt;

  StructForStmt(std::vector<Stmt *> loop_vars,
                SNode *snode,
                std::unique_ptr<Block> &&body,
                int vectorize,
                int parallelize)
      : loop_vars(loop_vars),
        snode(snode),
        body(std::move(body)),
        vectorize(vectorize),
        parallelize(parallelize) {
    block_size = 0;
  }

  bool is_container_statement() const override {
    return true;
  }

  DEFINE_ACCEPT
};

class WhileStmt : public Stmt {
 public:
  Stmt *mask;
  std::unique_ptr<Block> body;

  WhileStmt(std::unique_ptr<Block> &&body) : body(std::move(body)) {
  }

  bool is_container_statement() const override {
    return true;
  }

  DEFINE_ACCEPT
};

class FrontendWhileStmt : public Stmt {
 public:
  Expr cond;
  std::unique_ptr<Block> body;

  FrontendWhileStmt(Expr cond) : cond(load_if_ptr(cond)) {
  }

  bool is_container_statement() const override {
    return true;
  }

  DEFINE_ACCEPT
};

inline void IRBuilder::insert(std::unique_ptr<Stmt> &&stmt, int location) {
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

class EvalExpression : public Expression {
 public:
  Stmt *stmt_ptr;
  int stmt_id;
  EvalExpression(Stmt *stmt) : stmt_ptr(stmt), stmt_id(stmt_ptr->id) {
    // cache stmt->id since it may be released later
  }

  std::string serialize() override {
    return fmt::format("%{}", stmt_id);
  }

  void flatten(VecStatement &ret) override {
    stmt = stmt_ptr;
  }
};

class RangeAssumptionExpression : public Expression {
 public:
  Expr input, base;
  int low, high;

  RangeAssumptionExpression(const Expr &input,
                            const Expr &base,
                            int low,
                            int high)
      : input(input), base(base), low(low), high(high) {
  }

  std::string serialize() override {
    return fmt::format("assume_in_range({}{:+d} <= ({}) < {}{:+d})",
                       base.serialize(), low, input.serialize(),
                       base.serialize(), high);
  }

  void flatten(VecStatement &ret) override {
    input->flatten(ret);
    base->flatten(ret);
    ret.push_back(
        Stmt::make<RangeAssumptionStmt>(input->stmt, base->stmt, low, high));
    stmt = ret.back().get();
  }
};

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

  bool is_lvalue() const override {
    return true;
  }
};

class ProbeExpression : public Expression {
 public:
  SNode *snode;
  ExpressionGroup indices;
  ProbeExpression(SNode *snode, const ExpressionGroup &indices)
      : snode(snode), indices(indices) {
  }

  std::string serialize() override {
    return fmt::format("probe({}, [{}])", snode->node_type_name,
                       indices.serialize());
  }

  void flatten(VecStatement &ret) override {
    std::vector<Stmt *> indices_stmt;
    for (int i = 0; i < (int)indices.size(); i++) {
      indices[i]->flatten(ret);
      indices_stmt.push_back(indices[i]->stmt);
    }
    ret.push_back(std::make_unique<SNodeOpStmt>(SNodeOpType::probe, snode,
                                                indices_stmt, nullptr));
    stmt = ret.back().get();
  }
};

class GlobalLoadExpression : public Expression {
 public:
  Expr ptr;
  GlobalLoadExpression(Expr ptr) : ptr(ptr) {
  }

  std::string serialize() override {
    return "load " + ptr.serialize();
  }

  void flatten(VecStatement &ret) override {
    ptr->flatten(ret);
    ret.push_back(std::make_unique<GlobalLoadStmt>(ptr->stmt));
    stmt = ret.back().get();
  }
};

class ConstExpression : public Expression {
 public:
  TypedConstant val;

  template <typename T>
  ConstExpression(const T &x) : val(x) {
  }

  std::string serialize() override {
    return val.stringify();
  }

  void flatten(VecStatement &ret) override {
    ret.push_back(Stmt::make<ConstStmt>(val));
    stmt = ret.back().get();
  }
};

template <typename T, typename... Indices>
T &Expr::val(Indices... indices) {
  auto e = this->cast<GlobalVariableExpression>();
  TC_ASSERT(is<GlobalVariableExpression>());
  return *(T *)val_tmp(get_data_type<T>(), indices...);
}

inline Expr load(Expr ptr) {
  TC_ASSERT(ptr.is<GlobalPtrExpression>());
  return Expr::make<GlobalLoadExpression>(ptr);
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

inline void Cache(int v, const Expr &var) {
  dec.scratch_opt.push_back(std::make_pair(v, var.snode()));
}

inline void BlockDim(int v) {
  dec.block_size = v;
}

class PragmaSLPStmt : public Stmt {
 public:
  int slp_width;

  PragmaSLPStmt(int slp_width) : slp_width(slp_width) {
  }

  DEFINE_ACCEPT
};

class VectorElement {
 public:
  Stmt *stmt;
  int index;

  VectorElement() : stmt(nullptr), index(0) {
  }

  VectorElement(Stmt *stmt, int index) : stmt(stmt), index(index) {
  }
};

class ElementShuffleStmt : public Stmt {
 public:
  LaneAttribute<VectorElement> elements;

  ElementShuffleStmt(const LaneAttribute<VectorElement> &elements)
      : elements(elements) {
    width() = elements.size();
    element_type() = elements[0].stmt->element_type();
    for (int i = 0; i < width(); i++) {
      add_operand(this->elements[i].stmt);
    }
  }

  DEFINE_ACCEPT
};

inline void SLP(int v) {
  current_ast_builder().insert(Stmt::make<PragmaSLPStmt>(v));
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

  For(ExpressionGroup i, Expr global, const std::function<void()> &func) {
    auto stmt_unique = std::make_unique<FrontendForStmt>(i, global);
    auto stmt = stmt_unique.get();
    current_ast_builder().insert(std::move(stmt_unique));
    auto _ = current_ast_builder().create_scope(stmt->body);
    func();
  }

  For(Expr global, const std::function<void(Expr)> &func) {
    auto i = Expr(std::make_shared<IdExpression>());
    auto stmt_unique = std::make_unique<FrontendForStmt>(i, global);
    auto stmt = stmt_unique.get();
    current_ast_builder().insert(std::move(stmt_unique));
    auto _ = current_ast_builder().create_scope(stmt->body);
    func(i);
  }

  For(Expr global, const std::function<void(Expr, Expr)> &func) {
    auto i = Expr(std::make_shared<IdExpression>());
    auto j = Expr(std::make_shared<IdExpression>());
    auto stmt_unique = std::make_unique<FrontendForStmt>((i, j), global);
    auto stmt = stmt_unique.get();
    current_ast_builder().insert(std::move(stmt_unique));
    auto _ = current_ast_builder().create_scope(stmt->body);
    func(i, j);
  }

  For(Expr global, const std::function<void(Expr, Expr, Expr)> &func) {
    auto i = Expr(std::make_shared<IdExpression>());
    auto j = Expr(std::make_shared<IdExpression>());
    auto k = Expr(std::make_shared<IdExpression>());
    auto stmt_unique = std::make_unique<FrontendForStmt>((i, j, k), global);
    auto stmt = stmt_unique.get();
    current_ast_builder().insert(std::move(stmt_unique));
    auto _ = current_ast_builder().create_scope(stmt->body);
    func(i, j, k);
  }

  For(Expr global, const std::function<void(Expr, Expr, Expr, Expr)> &func) {
    auto i = Expr(std::make_shared<IdExpression>());
    auto j = Expr(std::make_shared<IdExpression>());
    auto k = Expr(std::make_shared<IdExpression>());
    auto l = Expr(std::make_shared<IdExpression>());
    auto stmt_unique = std::make_unique<FrontendForStmt>((i, j, k, l), global);
    auto stmt = stmt_unique.get();
    current_ast_builder().insert(std::move(stmt_unique));
    auto _ = current_ast_builder().create_scope(stmt->body);
    func(i, j, k, l);
  }

  For(Expr s, Expr e, const std::function<void(Expr)> &func);
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

inline Expr Var(Expr x) {
  auto var = Expr(std::make_shared<IdExpression>());
  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id, DataType::unknown));
  var = x;
  return var;
}

TLANG_NAMESPACE_END
