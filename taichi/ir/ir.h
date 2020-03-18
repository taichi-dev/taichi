// Intermediate representation system

#pragma once

#include <atomic>
#include <unordered_map>
#include "taichi/common/util.h"
#include "taichi/common/bit.h"
#include "taichi/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/expr.h"
#include "taichi/program/compile_config.h"
#include "taichi/llvm/llvm_fwd.h"

TLANG_NAMESPACE_BEGIN

class DiffRange {
 public:
 private:
  bool related;

 public:
  int coeff;
  int low, high;

  DiffRange() : DiffRange(false, 0) {
  }

  DiffRange(bool related, int coeff) : DiffRange(related, 0, 0) {
    TI_ASSERT(related == false);
  }

  DiffRange(bool related, int coeff, int low)
      : DiffRange(related, coeff, low, low + 1) {
  }

  DiffRange(bool related, int coeff, int low, int high)
      : related(related), coeff(coeff), low(low), high(high) {
    if (!related) {
      this->low = this->high = 0;
    }
  }

  bool related_() const {
    return related;
  }

  bool linear_related() const {
    return related && coeff == 1;
  }

  bool certain() {
    TI_ASSERT(related);
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
class ExprGroup;
class ScratchPads;

#define PER_STATEMENT(x) class x;
#include "taichi/inc/statements.inc.h"
#undef PER_STATEMENT

// IR passes
namespace irpass {

struct OffloadedResult {
  // Total size in bytes of the global temporary variables
  std::size_t total_size;
  // Offloaded local variables to its offset in the global tmps memory.
  std::unordered_map<const Stmt *, std::size_t> local_to_global_offset;
};

void re_id(IRNode *root);
void flag_access(IRNode *root);
void die(IRNode *root);
void simplify(IRNode *root);
void alg_simp(IRNode *root, const CompileConfig &config);
void full_simplify(IRNode *root, const CompileConfig &config);
void print(IRNode *root);
void lower(IRNode *root);
void typecheck(IRNode *root);
void loop_vectorize(IRNode *root);
void slp_vectorize(IRNode *root);
void vector_split(IRNode *root, int max_width, bool serial_schedule);
void replace_all_usages_with(IRNode *root, Stmt *old_stmt, Stmt *new_stmt);
void check_out_of_bound(IRNode *root);
void lower_access(IRNode *root, bool lower_atomic);
void make_adjoint(IRNode *root);
void constant_fold(IRNode *root);
OffloadedResult offload(IRNode *root);
void fix_block_parents(IRNode *root);
void replace_statements_with(IRNode *root,
                             std::function<bool(Stmt *)> filter,
                             std::function<std::unique_ptr<Stmt>()> generator);
void demote_dense_struct_fors(IRNode *root);
void demote_atomics(IRNode *root);
void reverse_segments(IRNode *root);  // for autograd
std::unique_ptr<ScratchPads> initialize_scratch_pad(StructForStmt *root);
std::vector<SNode *> gather_deactivations(IRNode *root);
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
  bool strictly_serialized;
  ScratchPadOptions scratch_opt;
  int block_dim;
  bool uniform;

  DecoratorRecorder() {
    reset();
  }

  void reset() {
    vectorize = -1;
    parallelize = 0;
    uniform = false;
    scratch_opt.clear();
    block_dim = 0;
    strictly_serialized = false;
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

  std::unique_ptr<ScopeGuard> create_scope(std::unique_ptr<Block> &list);

  Block *current_block() {
    if (stack.empty())
      return nullptr;
    else
      return stack.back();
  }

  Stmt *get_last_stmt();

  void stop_gradient(SNode *);
};

inline Expr load_if_ptr(const Expr &ptr);
inline Expr smart_load(const Expr &var);

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

  VecStatement(pStmt &&stmt) {
    push_back(std::move(stmt));
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
      TI_ERROR(
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
      TI_NOT_IMPLEMENTED;          \
  }

  DEFINE_VISIT(Block);
#define PER_STATEMENT(x) DEFINE_VISIT(x)
#include "taichi/inc/statements.inc.h"

#undef PER_STATEMENT
};

class IRNode {
 public:
  virtual void accept(IRVisitor *visitor) {
    TI_NOT_IMPLEMENTED
  }
  virtual ~IRNode() = default;
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
    TI_ASSERT(0 <= i && i < (int)data.size());
    return data[i];
  }

  const T &operator[](int i) const {
    TI_ASSERT(0 <= i && i < (int)data.size());
    return data[i];
  }

  LaneAttribute slice(int begin, int end) {
    return LaneAttribute(
        std::vector<T>(data.begin() + begin, data.begin() + end));
  }

  // for initializing single lane
  LaneAttribute &operator=(const T &t) {
    TI_ASSERT(data.size() == 1);
    data[0] = t;
    return *this;
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
      TI_P(bracket);
      TI_NOT_IMPLEMENTED
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
      TI_P(bracket);
      TI_NOT_IMPLEMENTED
    }
    return ret;
  }

  operator T() const {
    TI_ASSERT(data.size() == 1);
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
 protected:  // NOTE: operands should not be directly modified, for the
             // correctness of operand_bitmap
  std::vector<Stmt **> operands;

 public:
  static std::atomic<int> instance_id_counter;
  int instance_id;
  int id;
  Block *parent;
  uint64 operand_bitmap;
  bool erased;
  std::string tb;
  Stmt *adjoint;
  llvm::Value *value;
  bool is_ptr;

  Stmt(const Stmt &stmt) = delete;

  Stmt() {
    adjoint = nullptr;
    parent = nullptr;
    value = nullptr;
    instance_id = instance_id_counter++;
    id = instance_id;
    operand_bitmap = 0;
    erased = false;
    is_ptr = false;
  }

  static uint64 operand_hash(Stmt *stmt) {
    return uint64(1) << ((uint64(stmt) >> 4) % 64);
  }

  int &width() {
    return ret_type.width;
  }

  const int &width() const {
    return ret_type.width;
  }

  virtual bool is_container_statement() const {
    return false;
  }

  DataType &element_type() {
    return ret_type.data_type;
  }

  VectorType ret_type;

  std::string ret_data_type_name() const {
    return ret_type.str();
  }

  std::string type_hint() const {
    if (ret_type.data_type == DataType::unknown)
      return "";
    else
      return fmt::format("<{}> {}", ret_type.short_str(),
                         is_ptr ? "ptr " : " ");
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
    TI_ASSERT(is<T>());
    return dynamic_cast<T *>(this);
  }

  template <typename T>
  T *cast() {
    return dynamic_cast<T *>(this);
  }

  TI_FORCE_INLINE int num_operands() const {
    return (int)operands.size();
  }

  TI_FORCE_INLINE Stmt *operand(int i) const {
    // TI_ASSERT(0 <= i && i < (int)operands.size());
    return *operands[i];
  }

  std::vector<Stmt *> get_operands() const {
    std::vector<Stmt *> ret;
    for (int i = 0; i < num_operands(); i++) {
      ret.push_back(*operands[i]);
    }
    return ret;
  }

  void rebuild_operand_bitmap() {
    return;  // disable bitmap maintenance since the fact that the user can
             // modify the operand from the statement field (e.g.
             // IntegralOffsetStmt::input) makes it impossible to achieve our
             // goal
    operand_bitmap = 0;
    for (int i = 0; i < (int)operands.size(); i++) {
      operand_bitmap |= operand_hash(*operands[i]);
    }
  }

  void set_operand(int i, Stmt *stmt) {
    *operands[i] = stmt;
    rebuild_operand_bitmap();
  }

  void add_operand(Stmt *&stmt) {
    operands.push_back(&stmt);
    rebuild_operand_bitmap();
  }

  virtual void rebuild_operands() {
    TI_NOT_IMPLEMENTED;
  }

  TI_FORCE_INLINE bool may_have_operand(Stmt *stmt) const {
    return (operand_bitmap & operand_hash(stmt)) != 0;
  }

  void replace_with(Stmt *new_stmt);

  void replace_with(VecStatement &&new_statements, bool replace_usages = true);

  virtual void replace_operand_with(Stmt *old_stmt, Stmt *new_stmt);

  IRNode *get_ir_root();

  virtual void repeat(int factor) {
    ret_type.width *= factor;
  }

  // returns the inserted stmt
  Stmt *insert_before_me(std::unique_ptr<Stmt> &&new_stmt);

  // returns the inserted stmt
  Stmt *insert_after_me(std::unique_ptr<Stmt> &&new_stmt);

  virtual bool integral_operands() const {
    return true;
  }

  virtual bool has_global_side_effect() const {
    return true;
  }

  template <typename T, typename... Args>
  static pStmt make(Args &&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  static std::unique_ptr<T> make_typed(Args &&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
  }

  void infer_type() {
    irpass::typecheck(this);
  }

  void set_tb(std::string tb) {
    this->tb = tb;
  }

  std::string type();

  virtual ~Stmt() override = default;
};

// always a tree - used as rvalues
class Expression {
 public:
  Stmt *stmt;
  std::string tb;
  std::map<std::string, std::string> attributes;

  Expression() {
    stmt = nullptr;
  }

  virtual std::string serialize() = 0;

  virtual void flatten(VecStatement &ret) {
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

  std::string get_attribute(const std::string &key) const {
    if (auto it = attributes.find(key); it == attributes.end()) {
      TI_ERROR("Attribute {} not found.", key);
    } else {
      return it->second;
    }
  }
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

  ExprGroup(ExprGroup a, const Expr &b) {
    exprs = a.exprs;
    exprs.push_back(b);
  }

  ExprGroup(const Expr &a, ExprGroup b) {
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

  ExprGroup loaded() const;
};

inline ExprGroup operator,(const Expr &a, const Expr &b) {
  return ExprGroup(a, b);
}

inline ExprGroup operator,(const ExprGroup &a, const Expr &b) {
  return ExprGroup(a, b);
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

  virtual bool has_global_side_effect() const override {
    return false;
  }

  DEFINE_ACCEPT
};

// updates mask, break if no active
class WhileControlStmt : public Stmt {
 public:
  Stmt *mask;
  Stmt *cond;
  WhileControlStmt(Stmt *mask, Stmt *cond) : mask(mask), cond(cond) {
    add_operand(this->mask);
    add_operand(this->cond);
  }
  DEFINE_ACCEPT;
};

class UnaryOpStmt : public Stmt {
 public:
  UnaryOpType op_type;
  Stmt *operand;
  DataType cast_type;
  bool cast_by_value = true;

  UnaryOpStmt(UnaryOpType op_type, Stmt *operand)
      : op_type(op_type), operand(operand) {
    TI_ASSERT(!operand->is<AllocaStmt>());
    add_operand(this->operand);
    cast_type = DataType::unknown;
    cast_by_value = true;
  }

  bool same_operation(UnaryOpStmt *o) const {
    if (op_type == o->op_type) {
      if (op_type == UnaryOpType::cast) {
        return cast_type == o->cast_type;
      } else {
        return true;
      }
    }
    return false;
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }
  DEFINE_ACCEPT
};

class ArgLoadStmt : public Stmt {
 public:
  int arg_id;

  ArgLoadStmt(int arg_id, bool is_ptr = false) : arg_id(arg_id) {
    this->is_ptr = is_ptr;
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }

  DEFINE_ACCEPT
};

class ArgLoadExpression : public Expression {
 public:
  int arg_id;

  ArgLoadExpression(int arg_id) : arg_id(arg_id) {
  }

  std::string serialize() override {
    return fmt::format("arg[{}]", arg_id);
  }

  void flatten(VecStatement &ret) override {
    auto ran = std::make_unique<ArgLoadStmt>(arg_id);
    ret.push_back(std::move(ran));
    stmt = ret.back().get();
  }
};

// For return values
class FrontendArgStoreStmt : public Stmt {
 public:
  int arg_id;
  Expr expr;

  FrontendArgStoreStmt(int arg_id, Expr expr) : arg_id(arg_id), expr(expr) {
  }

  // Arguments are considered global (nonlocal)
  virtual bool has_global_side_effect() const override {
    return true;
  }

  DEFINE_ACCEPT
};

// For return values
class ArgStoreStmt : public Stmt {
 public:
  int arg_id;
  Stmt *val;

  ArgStoreStmt(int arg_id, Stmt *val) : arg_id(arg_id), val(val) {
    add_operand(this->val);
  }

  // Arguments are considered global (nonlocal)
  virtual bool has_global_side_effect() const override {
    return true;
  }

  DEFINE_ACCEPT
};

class RandStmt : public Stmt {
 public:
  RandStmt(DataType dt) {
    ret_type.data_type = dt;
  }

  virtual bool has_global_side_effect() const override {
    return false;
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
  UnaryOpType type;
  Expr operand;
  DataType cast_type;
  bool cast_by_value;

  UnaryOpExpression(UnaryOpType type, const Expr &operand)
      : type(type), operand(smart_load(operand)) {
    cast_type = DataType::unknown;
    cast_by_value = true;
  }

  std::string serialize() override {
    if (type == UnaryOpType::cast) {
      std::string reint = cast_by_value ? "" : "reinterpret_";
      return fmt::format("({}{}<{}> {})", reint, unary_op_type_name(type),
                         data_type_name(cast_type), operand->serialize());
    } else {
      return fmt::format("({} {})", unary_op_type_name(type),
                         operand->serialize());
    }
  }

  void flatten(VecStatement &ret) override {
    operand->flatten(ret);
    auto unary = std::make_unique<UnaryOpStmt>(type, operand->stmt);
    if (type == UnaryOpType::cast) {
      unary->cast_type = cast_type;
      unary->cast_by_value = cast_by_value;
    }
    stmt = unary.get();
    stmt->tb = tb;
    ret.push_back(std::move(unary));
  }
};

class BinaryOpStmt : public Stmt {
 public:
  BinaryOpType op_type;
  Stmt *lhs, *rhs;

  BinaryOpStmt(BinaryOpType op_type, Stmt *lhs, Stmt *rhs)
      : op_type(op_type), lhs(lhs), rhs(rhs) {
    TI_ASSERT(!lhs->is<AllocaStmt>());
    TI_ASSERT(!rhs->is<AllocaStmt>());
    add_operand(this->lhs);
    add_operand(this->rhs);
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }
  DEFINE_ACCEPT
};

class TernaryOpStmt : public Stmt {
 public:
  TernaryOpType op_type;
  Stmt *op1, *op2, *op3;

  TernaryOpStmt(TernaryOpType op_type, Stmt *op1, Stmt *op2, Stmt *op3)
      : op_type(op_type), op1(op1), op2(op2), op3(op3) {
    TI_ASSERT(!op1->is<AllocaStmt>());
    TI_ASSERT(!op2->is<AllocaStmt>());
    TI_ASSERT(!op3->is<AllocaStmt>());
    add_operand(this->op1);
    add_operand(this->op2);
    add_operand(this->op3);
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }
  DEFINE_ACCEPT
};

class AtomicOpStmt : public Stmt {
 public:
  AtomicOpType op_type;
  Stmt *dest, *val;

  AtomicOpStmt(AtomicOpType op_type, Stmt *dest, Stmt *val)
      : op_type(op_type), dest(dest), val(val) {
    add_operand(this->dest);
    add_operand(this->val);
  }

  DEFINE_ACCEPT
};

class BinaryOpExpression : public Expression {
 public:
  BinaryOpType type;
  Expr lhs, rhs;

  BinaryOpExpression(const BinaryOpType &type, const Expr &lhs, const Expr &rhs)
      : type(type) {
    this->lhs.set(smart_load(lhs));
    this->rhs.set(smart_load(rhs));
  }

  std::string serialize() override {
    return fmt::format("({} {} {})", lhs->serialize(),
                       binary_op_type_symbol(type), rhs->serialize());
  }

  void flatten(VecStatement &ret) override {
    // if (stmt)
    //  return;
    lhs->flatten(ret);
    rhs->flatten(ret);
    ret.push_back(std::make_unique<BinaryOpStmt>(type, lhs->stmt, rhs->stmt));
    ret.back()->tb = tb;
    stmt = ret.back().get();
  }
};

class TrinaryOpExpression : public Expression {
 public:
  TernaryOpType type;
  Expr op1, op2, op3;

  TrinaryOpExpression(TernaryOpType type,
                      const Expr &op1,
                      const Expr &op2,
                      const Expr &op3)
      : type(type) {
    this->op1.set(load_if_ptr(op1));
    this->op2.set(load_if_ptr(op2));
    this->op3.set(load_if_ptr(op3));
  }

  std::string serialize() override {
    return fmt::format("{}({} {} {})", ternary_type_name(type),
                       op1->serialize(), op2->serialize(), op3->serialize());
  }

  void flatten(VecStatement &ret) override {
    // if (stmt)
    //  return;
    op1->flatten(ret);
    op2->flatten(ret);
    op3->flatten(ret);
    ret.push_back(
        std::make_unique<TernaryOpStmt>(type, op1->stmt, op2->stmt, op3->stmt));
    stmt = ret.back().get();
  }
};

class ExternalPtrStmt : public Stmt {
 public:
  LaneAttribute<Stmt *> base_ptrs;
  std::vector<Stmt *> indices;
  bool activate;

  ExternalPtrStmt(const LaneAttribute<Stmt *> &base_ptrs,
                  const std::vector<Stmt *> &indices)
      : base_ptrs(base_ptrs), indices(indices) {
    DataType dt = DataType::f32;
    for (int i = 0; i < (int)base_ptrs.size(); i++) {
      TI_ASSERT(base_ptrs[i] != nullptr);
      TI_ASSERT(base_ptrs[i]->is<ArgLoadStmt>());
    }
    for (int i = 0; i < (int)base_ptrs.size(); i++) {
      add_operand(this->base_ptrs[i]);
    }
    for (int i = 0; i < (int)indices.size(); i++) {
      add_operand(this->indices[i]);
    }
    width() = base_ptrs.size();
    element_type() = dt;
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }

  DEFINE_ACCEPT
};

class GlobalPtrStmt : public Stmt {
 public:
  LaneAttribute<SNode *> snodes;
  std::vector<Stmt *> indices;
  bool activate;

  GlobalPtrStmt(const LaneAttribute<SNode *> &snodes,
                const std::vector<Stmt *> &indices,
                bool activate = true)
      : snodes(snodes), indices(indices), activate(activate) {
    for (int i = 0; i < (int)snodes.size(); i++) {
      TI_ASSERT(snodes[i] != nullptr);
      TI_ASSERT(snodes[0]->dt == snodes[i]->dt);
    }
    for (int i = 0; i < (int)indices.size(); i++) {
      add_operand(this->indices[i]);
    }
    width() = snodes.size();
    element_type() = snodes[0]->dt;
  }

  virtual bool has_global_side_effect() const override {
    return activate;
  }

  DEFINE_ACCEPT
};

class ExternalTensorExpression : public Expression {
 public:
  DataType dt;
  int dim;
  int arg_id;

  ExternalTensorExpression(const DataType &dt, int dim, int arg_id)
      : dt(dt), dim(dim), arg_id(arg_id) {
    set_attribute("dim", std::to_string(dim));
  }

  std::string serialize() override {
    return fmt::format("{}d_ext_arr", dim);
  }

  void flatten(VecStatement &ret) override {
    auto ptr = Stmt::make<ArgLoadStmt>(arg_id, true);
    ret.push_back(std::move(ptr));
    stmt = ret.back().get();
  }
};

class GlobalVariableExpression : public Expression {
 public:
  Identifier ident;
  DataType dt;
  SNode *snode;
  bool has_ambient;
  TypedConstant ambient_value;
  bool is_primal;
  Expr adjoint;

  GlobalVariableExpression(DataType dt, Ident ident) : ident(ident), dt(dt) {
    snode = nullptr;
    has_ambient = false;
    is_primal = true;
  }

  GlobalVariableExpression(SNode *snode) : snode(snode) {
    dt = snode->dt;
    has_ambient = false;
    is_primal = true;
  }

  void set_snode(SNode *snode) {
    this->snode = snode;
    set_attribute("dim", std::to_string(snode->num_active_indices));
  }

  std::string serialize() override {
    return "#" + ident.name();
  }

  void flatten(VecStatement &ret) override {
    TI_ASSERT(snode->num_active_indices == 0);
    auto ptr = Stmt::make<GlobalPtrStmt>(LaneAttribute<SNode *>(snode),
                                         std::vector<Stmt *>());
    ret.push_back(std::move(ptr));
  }
};

class GlobalPtrExpression : public Expression {
 public:
  Expr var;
  ExprGroup indices;

  GlobalPtrExpression(const Expr &var, const ExprGroup &indices)
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
    if (var.is<GlobalVariableExpression>()) {
      ret.push_back(std::make_unique<GlobalPtrStmt>(
          var.cast<GlobalVariableExpression>()->snode, index_stmts));
    } else {
      TI_ASSERT(var.is<ExternalTensorExpression>());
      var->flatten(ret);
      ret.push_back(std::make_unique<ExternalPtrStmt>(
          var.cast<ExternalTensorExpression>()->stmt, index_stmts));
    }
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

// Value cast
Expr cast(const Expr &input, DataType dt);

template <typename T>
Expr cast(const Expr &input) {
  return taichi::lang::cast(input, get_data_type<T>());
}

Expr bit_cast(const Expr &input, DataType dt);

template <typename T>
Expr bit_cast(const Expr &input) {
  return taichi::lang::bit_cast(input, get_data_type<T>());
}

class Block : public IRNode {
 public:
  Block *parent;
  std::vector<std::unique_ptr<Stmt>> statements, trash_bin;
  std::map<Ident, Stmt *> local_var_alloca;
  Stmt *mask_var;
  std::vector<SNode *> stop_gradients;

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
    return -1;
  }

  void erase(int location);

  void erase(Stmt *stmt);

  void insert(std::unique_ptr<Stmt> &&stmt, int location = -1);

  void insert(VecStatement &&stmt, int location = -1);

  void replace_statements_in_range(int start, int end, VecStatement &&stmts);

  void set_statements(VecStatement &&stmts) {
    statements.clear();
    for (int i = 0; i < (int)stmts.size(); i++) {
      insert(std::move(stmts[i]), i);
    }
  }

  void replace_with(Stmt *old_statement, std::unique_ptr<Stmt> &&new_statement);

  void insert_before(Stmt *old_statement, VecStatement &&new_statements) {
    int location = -1;
    for (int i = 0; i < (int)statements.size(); i++) {
      if (old_statement == statements[i].get()) {
        location = i;
        break;
      }
    }
    TI_ASSERT(location != -1);
    for (int i = (int)new_statements.size() - 1; i >= 0; i--) {
      insert(std::move(new_statements[i]), location);
    }
  }

  void replace_with(Stmt *old_statement,
                    VecStatement &&new_statements,
                    bool replace_usages = true) {
    int location = -1;
    for (int i = 0; i < (int)statements.size(); i++) {
      if (old_statement == statements[i].get()) {
        location = i;
        break;
      }
    }
    TI_ASSERT(location != -1);
    if (replace_usages)
      old_statement->replace_with(new_statements.back().get());
    trash_bin.push_back(std::move(statements[location]));
    statements.erase(statements.begin() + location);
    for (int i = (int)new_statements.size() - 1; i >= 0; i--) {
      insert(std::move(new_statements[i]), location);
    }
  }

  Stmt *lookup_var(Ident ident) const;

  Stmt *mask();

  Stmt *back() const {
    return statements.back().get();
  }

  template <typename T, typename... Args>
  Stmt *push_back(Args &&... args) {
    auto stmt = std::make_unique<T>(std::forward<Args>(args)...);
    stmt->parent = this;
    statements.emplace_back(std::move(stmt));
    return back();
  }

  std::size_t size() {
    return statements.size();
  }

  pStmt &operator[](int i) {
    return statements[i];
  }

  DEFINE_ACCEPT
};

class FrontendAtomicStmt : public Stmt {
 public:
  AtomicOpType op_type;
  Expr dest, val;

  FrontendAtomicStmt(AtomicOpType op_type, Expr dest, Expr val);

  DEFINE_ACCEPT
};

class FrontendSNodeOpStmt : public Stmt {
 public:
  SNodeOpType op_type;
  SNode *snode;
  ExprGroup indices;
  Expr val;

  FrontendSNodeOpStmt(SNodeOpType op_type,
                      SNode *snode,
                      ExprGroup indices,
                      Expr val = Expr(nullptr))
      : op_type(op_type), snode(snode), indices(indices.loaded()), val(val) {
    if (val.expr != nullptr) {
      TI_ASSERT(op_type == SNodeOpType::append);
      this->val.set(load_if_ptr(val));
    } else {
      TI_ASSERT(op_type != SNodeOpType::append);
    }
  }

  DEFINE_ACCEPT
};

class SNodeOpStmt : public Stmt {
 public:
  SNodeOpType op_type;
  SNode *snode;
  Stmt *ptr;
  Stmt *val;
  std::vector<Stmt *> indices;

  SNodeOpStmt(SNodeOpType op_type, SNode *snode, Stmt *ptr, Stmt *val = nullptr)
      : op_type(op_type), snode(snode), ptr(ptr), val(val) {
    add_operand(this->ptr);
    if (val)
      add_operand(this->val);
    width() = 1;
    element_type() = DataType::i32;
  }

  SNodeOpStmt(SNodeOpType op_type, SNode *snode, std::vector<Stmt *> indices)
      : op_type(op_type), snode(snode), indices(indices) {
    ptr = nullptr;
    val = nullptr;
    TI_ASSERT(op_type == SNodeOpType::is_active ||
              op_type == SNodeOpType::deactivate);
    add_operand(this->ptr);
    for (int i = 0; i < (int)indices.size(); i++) {
      add_operand(this->indices[i]);
    }
    width() = 1;
    element_type() = DataType::i32;
  }

  static bool activation_related(SNodeOpType op) {
    return op == SNodeOpType::activate || op == SNodeOpType::deactivate ||
           op == SNodeOpType::is_active;
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
  Stmt *cond;
  std::string text;
  std::vector<Stmt *> args;

  AssertStmt(const std::string &text, Stmt *cond) : cond(cond), text(text) {
    add_operand(this->cond);
    TI_ASSERT(cond);
  }

  AssertStmt(Stmt *cond,
             const std::string &text,
             const std::vector<Stmt *> &args)
      : cond(cond), text(text), args(args) {
    add_operand(this->cond);
    for (auto &arg : this->args) {
      add_operand(arg);
    }
    TI_ASSERT(cond);
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

  FrontendAssignStmt(const Expr &lhs, const Expr &rhs);

  DEFINE_ACCEPT
};

class GlobalLoadStmt : public Stmt {
 public:
  Stmt *ptr;

  GlobalLoadStmt(Stmt *ptr) : ptr(ptr) {
    add_operand(this->ptr);
  }

  virtual bool has_global_side_effect() const override {
    return false;
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

  void rebuild_operands() override {
    operands.clear();
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

  bool has_source(Stmt *alloca) const {
    for (int i = 0; i < width(); i++) {
      if (ptr[i].var == alloca)
        return true;
    }
    return false;
  }

  bool integral_operands() const override {
    return false;
  }

  Stmt *previous_store_or_alloca_in_block();

  virtual bool has_global_side_effect() const override {
    return false;
  }

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
      TI_ASSERT(val[0].dt == val[i].dt);
    }
  }

  void repeat(int factor) override {
    Stmt::repeat(factor);
    val.repeat(factor);
  }

  virtual bool has_global_side_effect() const override {
    return false;
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
  bool strictly_serialized;
  ScratchPadOptions scratch_opt;
  int block_dim;

  bool is_ranged() const {
    if (global_var.expr == nullptr) {
      return true;
    } else {
      return false;
    }
  }

  FrontendForStmt(const ExprGroup &loop_var, const Expr &global_var);

  FrontendForStmt(const Expr &loop_var, const Expr &begin, const Expr &end);

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
  bool reversed;
  int vectorize;
  int parallelize;
  int block_dim;
  bool strictly_serialized;

  RangeForStmt(Stmt *loop_var,
               Stmt *begin,
               Stmt *end,
               std::unique_ptr<Block> &&body,
               int vectorize,
               int parallelize,
               int block_dim,
               bool strictly_serialized)
      : loop_var(loop_var),
        begin(begin),
        end(end),
        body(std::move(body)),
        vectorize(vectorize),
        parallelize(parallelize),
        block_dim(block_dim),
        strictly_serialized(strictly_serialized) {
    reversed = false;
    add_operand(this->loop_var);
    add_operand(this->begin);
    add_operand(this->end);
  }

  bool is_container_statement() const override {
    return true;
  }

  void reverse() {
    reversed = !reversed;
  }

  DEFINE_ACCEPT
};

// for stmt over a structural node
class StructForStmt : public Stmt {
 public:
  std::vector<Stmt *> loop_vars;
  SNode *snode;
  std::unique_ptr<Block> body;
  std::unique_ptr<Block> block_initialization;
  std::unique_ptr<Block> block_finalization;
  int vectorize;
  int parallelize;
  int block_dim;
  ScratchPadOptions scratch_opt;

  StructForStmt(std::vector<Stmt *> loop_vars,
                SNode *snode,
                std::unique_ptr<Block> &&body,
                int vectorize,
                int parallelize,
                int block_dim)
      : loop_vars(loop_vars),
        snode(snode),
        body(std::move(body)),
        vectorize(vectorize),
        parallelize(parallelize),
        block_dim(block_dim) {
    for (auto &v : this->loop_vars) {
      add_operand(v);
    }
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

class FrontendBreakStmt : public Stmt {
 public:
  FrontendBreakStmt() {
  }

  bool is_container_statement() const override {
    return false;
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

#define Print(x) Print_(x, #x);

void Print_(const Expr &a, std::string str);

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

  bool is_lvalue() const override {
    return true;
  }
};

// This is just a wrapper class of FrontendAtomicStmt, so that we can turn
// ti.atomic_op() into an expression (with side effect).
class AtomicOpExpression : public Expression {
  // TODO(issue#332): Flatten this into AtomicOpStmt directly, then we can
  // deprecate FrontendAtomicStmt.
 public:
  AtomicOpType op_type;
  Expr dest, val;

  AtomicOpExpression(AtomicOpType op_type, Expr dest, Expr val)
      : op_type(op_type), dest(dest), val(val) {
  }

  std::string serialize() override {
    if (op_type == AtomicOpType::add) {
      return fmt::format("atomic_add({}, {})", dest.serialize(),
                         val.serialize());
    } else if (op_type == AtomicOpType::sub) {
      return fmt::format("atomic_sub({}, {})", dest.serialize(),
                         val.serialize());
    } else {
      // min/max not supported in the LLVM backend yet.
      TI_NOT_IMPLEMENTED;
    }
  }

  void flatten(VecStatement &ret) override {
    // FrontendAtomicStmt is the correct place to flatten sub-exprs like |dest|
    // and |val| (See LowerAST). This class only wraps the frontend atomic_op()
    // stmt as an expression.
    ret.push_back<FrontendAtomicStmt>(op_type, dest, val);
    stmt = ret.back().get();
  }
};

class SNodeOpExpression : public Expression {
 public:
  SNode *snode;
  SNodeOpType op_type;
  ExprGroup indices;
  Expr value;

  SNodeOpExpression(SNode *snode, SNodeOpType op_type, const ExprGroup &indices)
      : snode(snode), op_type(op_type), indices(indices) {
  }

  SNodeOpExpression(SNode *snode,
                    SNodeOpType op_type,
                    const ExprGroup &indices,
                    const Expr &value)
      : snode(snode), op_type(op_type), indices(indices), value(value) {
  }

  std::string serialize() override {
    if (value.expr) {
      return fmt::format("{}({}, [{}], {})", snode_op_type_name(op_type),
                         snode->get_node_type_name_hinted(),
                         indices.serialize(), value.serialize());
    } else {
      return fmt::format("{}({}, [{}])", snode_op_type_name(op_type),
                         snode->get_node_type_name_hinted(),
                         indices.serialize());
    }
  }

  void flatten(VecStatement &ret) override {
    std::vector<Stmt *> indices_stmt;
    for (int i = 0; i < (int)indices.size(); i++) {
      indices[i]->flatten(ret);
      indices_stmt.push_back(indices[i]->stmt);
    }
    if (op_type == SNodeOpType::is_active) {
      // is_active cannot be lowered all the way to a global pointer.
      // It should be lowered into a pointer to parent and an index.
      TI_ERROR_IF(
          snode->type != SNodeType::pointer && snode->type != SNodeType::hash,
          "ti.is_active only works on hash and pointer nodes.");
      ret.push_back<SNodeOpStmt>(SNodeOpType::is_active, snode, indices_stmt);
    } else {
      auto ptr = ret.push_back<GlobalPtrStmt>(snode, indices_stmt);
      if (op_type == SNodeOpType::append) {
        value->flatten(ret);
        ret.push_back<SNodeOpStmt>(SNodeOpType::append, snode, ptr,
                                   ret.back().get());
        TI_ERROR_IF(snode->type != SNodeType::dynamic,
                    "ti.append only works on dynamic nodes.");
        TI_ERROR_IF(snode->ch.size() != 1,
                    "ti.append only works on single-child dynamic nodes.");
        TI_ERROR_IF(data_type_size(snode->ch[0]->dt) != 4,
                    "ti.append only works on i32/f32 nodes.");
      } else if (op_type == SNodeOpType::length) {
        ret.push_back<SNodeOpStmt>(SNodeOpType::length, snode, ptr, nullptr);
      }
    }
    stmt = ret.back().get();
  }
};

class GlobalLoadExpression : public Expression {
 public:
  Expr ptr;
  GlobalLoadExpression(Expr ptr) : ptr(ptr) {
  }

  std::string serialize() override {
    return "gbl load " + ptr.serialize();
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

inline Expr load(Expr ptr) {
  TI_ASSERT(ptr.is<GlobalPtrExpression>());
  return Expr::make<GlobalLoadExpression>(ptr);
}

inline Expr load_if_ptr(const Expr &ptr) {
  if (ptr.is<GlobalPtrExpression>()) {
    return load(ptr);
  } else if (ptr.is<GlobalVariableExpression>()) {
    TI_ASSERT(ptr.cast<GlobalVariableExpression>()->snode->num_active_indices ==
              0);
    return load(ptr[ExprGroup()]);
  } else
    return ptr;
}

inline Expr ptr_if_global(const Expr &var) {
  if (var.is<GlobalVariableExpression>()) {
    // singleton global variable
    TI_ASSERT(var.snode()->num_active_indices == 0);
    return var[ExprGroup()];
  } else {
    // may be any local or global expr
    return var;
  }
}

inline Expr smart_load(const Expr &var) {
  return load_if_ptr(ptr_if_global(var));
}

extern DecoratorRecorder dec;

inline void Vectorize(int v) {
  dec.vectorize = v;
}

inline void Parallelize(int v) {
  dec.parallelize = v;
}

inline void StrictlySerialize() {
  dec.strictly_serialized = true;
}

inline void Cache(int v, const Expr &var) {
  dec.scratch_opt.push_back(std::make_pair(v, var.snode()));
}

inline void CacheL1(const Expr &var) {
  dec.scratch_opt.push_back(std::make_pair(1, var.snode()));
}

inline void BlockDim(int v) {
  TI_ASSERT(bit::is_power_of_two(v));
  dec.block_dim = v;
}

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

  For(ExprGroup i, Expr global, const std::function<void()> &func) {
    auto stmt_unique = std::make_unique<FrontendForStmt>(i, global);
    auto stmt = stmt_unique.get();
    current_ast_builder().insert(std::move(stmt_unique));
    auto _ = current_ast_builder().create_scope(stmt->body);
    func();
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

Expr Var(Expr x);

TLANG_NAMESPACE_END

#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
