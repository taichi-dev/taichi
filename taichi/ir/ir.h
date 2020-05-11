// Intermediate representation system

#pragma once

#include <atomic>
#include <unordered_set>
#include <unordered_map>
#include "taichi/common/util.h"
#include "taichi/common/bit.h"
#include "taichi/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/expr.h"
#include "taichi/program/compile_config.h"
#include "taichi/llvm/llvm_fwd.h"
#include "taichi/util/short_name.h"

TLANG_NAMESPACE_BEGIN

class DiffRange {
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
void simplify(IRNode *root, Kernel *kernel = nullptr);
void alg_simp(IRNode *root, const CompileConfig &config);
void whole_kernel_cse(IRNode *root);
void variable_optimization(IRNode *root, bool after_lower_access);
void extract_constant(IRNode *root);
void full_simplify(IRNode *root,
                   const CompileConfig &config,
                   Kernel *kernel = nullptr);
void print(IRNode *root, std::string *output = nullptr);
void lower(IRNode *root);
void convert_into_loop_index(IRNode *root);
void typecheck(IRNode *root, Kernel *kernel = nullptr);
void loop_vectorize(IRNode *root);
void slp_vectorize(IRNode *root);
void vector_split(IRNode *root, int max_width, bool serial_schedule);
void replace_all_usages_with(IRNode *root, Stmt *old_stmt, Stmt *new_stmt);
void check_out_of_bound(IRNode *root);
void lower_access(IRNode *root, bool lower_atomic, Kernel *kernel = nullptr);
void make_adjoint(IRNode *root, bool use_stack = false);
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
void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         bool vectorize,
                         bool grad,
                         bool ad_use_stack,
                         bool verbose,
                         bool lower_global_access = true);

// Analysis
namespace analysis {
void check_fields_registered(IRNode *root);
int count_statements(IRNode *root);
std::unordered_set<Stmt *> detect_fors_with_break(IRNode *root);
std::unordered_set<Stmt *> detect_loops_with_continue(IRNode *root);
std::unordered_set<SNode *> gather_deactivations(IRNode *root);
std::vector<Stmt *> gather_statements(IRNode *root,
                                      const std::function<bool(Stmt *)> &test);
std::unique_ptr<std::unordered_set<AtomicOpStmt *>> gather_used_atomics(
    IRNode *root);
bool has_store_or_atomic(IRNode *root, const std::vector<Stmt *> &vars);
std::pair<bool, Stmt *> last_store_or_atomic(IRNode *root, Stmt *var);
bool same_statements(IRNode *root1, IRNode *root2);
DiffRange value_diff(Stmt *stmt, int lane, Stmt *alloca);
void verify(IRNode *root);
}  // namespace analysis

}  // namespace irpass

IRBuilder &current_ast_builder();

bool maybe_same_address(Stmt *var1, Stmt *var2);

struct VectorType {
 private:
  bool _is_pointer;

 public:
  int width;
  DataType data_type;

  VectorType(int width, DataType data_type, bool is_pointer = false)
      : _is_pointer(is_pointer), width(width), data_type(data_type) {
  }

  VectorType() : _is_pointer(false), width(1), data_type(DataType::unknown) {
  }

  bool operator==(const VectorType &o) const {
    return width == o.width && data_type == o.data_type;
  }

  bool operator!=(const VectorType &o) const {
    return !(*this == o);
  }

  std::string pointer_suffix() const;
  std::string element_type_name() const;
  std::string str() const;

  bool is_pointer() const {
    return _is_pointer;
  }

  void set_is_pointer(bool v) {
    _is_pointer = v;
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

  void reset();
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
  Block *current_block();
  Stmt *get_last_stmt();
  void stop_gradient(SNode *);
};

Expr load_if_ptr(const Expr &ptr);
Expr load(const Expr &ptr);
Expr ptr_if_global(const Expr &var);

inline Expr smart_load(const Expr &var) {
  return load_if_ptr(ptr_if_global(var));
}

class Identifier {
 public:
  static int id_counter;
  std::string name_;

  int id;

  // Multiple identifiers can share the same name but must have different id's
  Identifier(const std::string &name_ = "") : name_(name_) {
    id = id_counter++;
  }

  std::string raw_name() const;

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

  VecStatement(std::vector<pStmt> &&other_stmts) {
    stmts = std::move(other_stmts);
  }

  Stmt *push_back(pStmt &&stmt);

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

  LaneAttribute(const T &t) : data(1, t) {
  }

  void resize(int s) {
    data.resize(s);
  }

  void reserve(int s) {
    data.reserve(s);
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

class StmtField {
 public:
  StmtField() = default;

  virtual bool equal(const StmtField *other) const = 0;

  virtual ~StmtField() = default;
};

template <typename T>
class StmtFieldNumeric final : public StmtField {
 private:
  T value;

 public:
  explicit StmtFieldNumeric(T value) : value(value) {
  }

  bool equal(const StmtField *other_generic) const override {
    if (auto other = dynamic_cast<const StmtFieldNumeric *>(other_generic)) {
      return other->value == value;
    } else {
      // Different types
      return false;
    }
  }
};

class StmtFieldSNode final : public StmtField {
 private:
  SNode *const &snode;

 public:
  explicit StmtFieldSNode(SNode *const &snode) : snode(snode) {
  }

  static int get_snode_id(SNode *snode);

  bool equal(const StmtField *other_generic) const override;
};

class StmtFieldManager {
 private:
  Stmt *stmt;

 public:
  std::vector<std::unique_ptr<StmtField>> fields;

  StmtFieldManager(Stmt *stmt) : stmt(stmt) {
  }

  template <typename T>
  void operator()(const char *key, T &&value);

  template <typename T, typename... Args>
  void operator()(const char *key_, T &&t, Args &&... rest) {
    std::string key(key_);
    size_t pos = key.find(',');
    std::string first_name = key.substr(0, pos);
    std::string rest_names =
        key.substr(pos + 2, int(key.size()) - (int)pos - 2);
    this->operator()(first_name.c_str(), std::forward<T>(t));
    this->operator()(rest_names.c_str(), std::forward<Args>(rest)...);
  }

  bool equal(StmtFieldManager &other) const;
};

#define TI_STMT_DEF_FIELDS(...) TI_IO_DEF(__VA_ARGS__)
#define TI_STMT_REG_FIELDS  \
  mark_fields_registered(); \
  io(field_manager)

class Stmt : public IRNode {
 protected:  // NOTE: operands should not be directly modified, for the
             // correctness of operand_bitmap
  std::vector<Stmt **> operands;

 public:
  StmtFieldManager field_manager;
  static std::atomic<int> instance_id_counter;
  int instance_id;
  int id;
  Block *parent;
  uint64 operand_bitmap;
  bool erased;
  bool fields_registered;
  std::string tb;
  bool is_ptr;
  VectorType ret_type;

  Stmt(const Stmt &stmt) = delete;
  Stmt();

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

  std::string ret_data_type_name() const {
    return ret_type.str();
  }

  std::string type_hint() const;

  std::string name() const {
    return fmt::format("${}", id);
  }

  std::string short_name() const {
    return make_short_name_by_id(id);
  }

  std::string raw_name() const {
    return fmt::format("tmp{}", id);
  }

  TI_FORCE_INLINE int num_operands() const {
    return (int)operands.size();
  }

  TI_FORCE_INLINE Stmt *operand(int i) const {
    // TI_ASSERT(0 <= i && i < (int)operands.size());
    return *operands[i];
  }

  std::vector<Stmt *> get_operands() const;

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

  void set_operand(int i, Stmt *stmt);
  void register_operand(Stmt *&stmt);
  int locate_operand(Stmt **stmt);
  void mark_fields_registered();

  virtual void rebuild_operands() {
    TI_NOT_IMPLEMENTED;
  }

  TI_FORCE_INLINE bool may_have_operand(Stmt *stmt) const {
    return (operand_bitmap & operand_hash(stmt)) != 0;
  }

  bool have_operand(Stmt *stmt) const;

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

  void set_tb(const std::string &tb) {
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

  struct FlattenContext {
    VecStatement stmts;
    Block *current_block = nullptr;

    inline Stmt *push_back(pStmt &&stmt) {
      return stmts.push_back(std::move(stmt));
    }

    template <typename T, typename... Args>
    T *push_back(Args &&... args) {
      return stmts.push_back<T>(std::forward<Args>(args)...);
    }

    Stmt *back_stmt() {
      return stmts.back().get();
    }
  };

  Expression() {
    stmt = nullptr;
  }

  virtual std::string serialize() = 0;

  virtual void flatten(FlattenContext *ctx) {
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

  std::string get_attribute(const std::string &key) const;
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

  ExprGroup(const ExprGroup &a, const Expr &b) {
    exprs = a.exprs;
    exprs.push_back(b);
  }

  ExprGroup(const Expr &a, const ExprGroup &b) {
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

  std::string serialize() const;
  ExprGroup loaded() const;
};

inline ExprGroup operator,(const Expr &a, const Expr &b) {
  return ExprGroup(a, b);
}

inline ExprGroup operator,(const ExprGroup &a, const Expr &b) {
  return ExprGroup(a, b);
}

class AllocaStmt : public Stmt {
 public:
  AllocaStmt(DataType type) {
    ret_type = VectorType(1, type);
    TI_STMT_REG_FIELDS;
  }

  AllocaStmt(int width, DataType type) {
    ret_type = VectorType(width, type);
    TI_STMT_REG_FIELDS;
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type);
  DEFINE_ACCEPT
};

// updates mask, break if no active
class WhileControlStmt : public Stmt {
 public:
  Stmt *mask;
  Stmt *cond;
  WhileControlStmt(Stmt *mask, Stmt *cond) : mask(mask), cond(cond) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(mask, cond);
  DEFINE_ACCEPT;
};

class ContinueStmt : public Stmt {
 public:
  // This is the loop on which this continue stmt has effects. It can be either
  // an offloaded task, or a for/while loop inside the kernel.
  Stmt *scope;

  ContinueStmt() : scope(nullptr) {
    TI_STMT_REG_FIELDS;
  }

  // For top-level loops, since they are parallelized to multiple threads (on
  // either CPU or GPU), `continue` becomes semantically equivalent to `return`.
  //
  // Caveat:
  // We should wrap each backend's kernel body into a function (as LLVM does).
  // The reason is that, each thread may handle more than one element,
  // depending on the backend's implementation.
  //
  // For example, CUDA uses gride-stride loops, the snippet below illustrates
  // the idea:
  //
  // __global__ foo_kernel(...) {
  //   for (int i = lower; i < upper; i += gridDim) {
  //     auto coord = compute_coords(i);
  //     // run_foo_kernel is produced by codegen
  //     run_foo_kernel(coord);
  //   }
  // }
  //
  // If run_foo_kernel() is directly inlined within foo_kernel(), `return`
  // could prematurely terminate the entire kernel.
  bool as_return() const;

  TI_STMT_DEF_FIELDS(scope);
  DEFINE_ACCEPT;
};

class UnaryOpStmt : public Stmt {
 public:
  UnaryOpType op_type;
  Stmt *operand;
  DataType cast_type;

  UnaryOpStmt(UnaryOpType op_type, Stmt *operand);

  bool same_operation(UnaryOpStmt *o) const;
  bool is_cast() const;

  virtual bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, op_type, operand, cast_type);
  DEFINE_ACCEPT
};

class ArgLoadStmt : public Stmt {
 public:
  int arg_id;

  ArgLoadStmt(int arg_id, bool is_ptr = false) : arg_id(arg_id) {
    this->is_ptr = is_ptr;
    TI_STMT_REG_FIELDS;
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, arg_id, is_ptr);
  DEFINE_ACCEPT
};

class RandStmt : public Stmt {
 public:
  RandStmt(DataType dt) {
    ret_type.data_type = dt;
    TI_STMT_REG_FIELDS;
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type);
  DEFINE_ACCEPT
};

class BinaryOpStmt : public Stmt {
 public:
  BinaryOpType op_type;
  Stmt *lhs, *rhs;

  BinaryOpStmt(BinaryOpType op_type, Stmt *lhs, Stmt *rhs)
      : op_type(op_type), lhs(lhs), rhs(rhs) {
    TI_ASSERT(!lhs->is<AllocaStmt>());
    TI_ASSERT(!rhs->is<AllocaStmt>());
    TI_STMT_REG_FIELDS;
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, op_type, lhs, rhs);
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
    TI_STMT_REG_FIELDS;
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, op1, op2, op3);
  DEFINE_ACCEPT
};

class AtomicOpStmt : public Stmt {
 public:
  AtomicOpType op_type;
  Stmt *dest, *val;

  AtomicOpStmt(AtomicOpType op_type, Stmt *dest, Stmt *val)
      : op_type(op_type), dest(dest), val(val) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(ret_type, op_type, dest, val);
  DEFINE_ACCEPT
};

class ExternalPtrStmt : public Stmt {
 public:
  LaneAttribute<Stmt *> base_ptrs;
  std::vector<Stmt *> indices;
  bool activate;

  ExternalPtrStmt(const LaneAttribute<Stmt *> &base_ptrs,
                  const std::vector<Stmt *> &indices);

  virtual bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, base_ptrs, indices, activate);
  DEFINE_ACCEPT
};

class GlobalPtrStmt : public Stmt {
 public:
  LaneAttribute<SNode *> snodes;
  std::vector<Stmt *> indices;
  bool activate;

  GlobalPtrStmt(const LaneAttribute<SNode *> &snodes,
                const std::vector<Stmt *> &indices,
                bool activate = true);

  virtual bool has_global_side_effect() const override {
    return activate;
  }

  TI_STMT_DEF_FIELDS(ret_type, snodes, indices, activate);
  DEFINE_ACCEPT
};

#include "expression.h"

class Block : public IRNode {
 public:
  Block *parent;
  std::vector<std::unique_ptr<Stmt>> statements, trash_bin;
  std::map<Identifier, Stmt *> local_var_alloca;
  Stmt *mask_var;
  std::vector<SNode *> stop_gradients;

  Block() {
    mask_var = nullptr;
    parent = nullptr;
  }

  bool has_container_statements();
  int locate(Stmt *stmt);
  void erase(int location);
  void erase(Stmt *stmt);
  std::unique_ptr<Stmt> extract(int location);
  std::unique_ptr<Stmt> extract(Stmt *stmt);
  void insert(std::unique_ptr<Stmt> &&stmt, int location = -1);
  void insert(VecStatement &&stmt, int location = -1);
  void replace_statements_in_range(int start, int end, VecStatement &&stmts);
  void set_statements(VecStatement &&stmts);
  void replace_with(Stmt *old_statement, std::unique_ptr<Stmt> &&new_statement);
  void insert_before(Stmt *old_statement, VecStatement &&new_statements);
  void replace_with(Stmt *old_statement,
                    VecStatement &&new_statements,
                    bool replace_usages = true);
  Stmt *lookup_var(const Identifier &ident) const;
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

class SNodeOpStmt : public Stmt {
 public:
  SNodeOpType op_type;
  SNode *snode;
  Stmt *ptr;
  Stmt *val;
  std::vector<Stmt *> indices;

  SNodeOpStmt(SNodeOpType op_type,
              SNode *snode,
              Stmt *ptr,
              Stmt *val = nullptr);

  SNodeOpStmt(SNodeOpType op_type,
              SNode *snode,
              const std::vector<Stmt *> &indices);

  static bool activation_related(SNodeOpType op) {
    return op == SNodeOpType::activate || op == SNodeOpType::deactivate ||
           op == SNodeOpType::is_active;
  }

  TI_STMT_DEF_FIELDS(ret_type, op_type, snode, ptr, val, indices);
  DEFINE_ACCEPT
};

class AssertStmt : public Stmt {
 public:
  Stmt *cond;
  std::string text;
  std::vector<Stmt *> args;

  AssertStmt(const std::string &text, Stmt *cond) : cond(cond), text(text) {
    TI_ASSERT(cond);
    TI_STMT_REG_FIELDS;
  }

  AssertStmt(Stmt *cond,
             const std::string &text,
             const std::vector<Stmt *> &args)
      : cond(cond), text(text), args(args) {
    TI_ASSERT(cond);
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(cond, text, args);
  DEFINE_ACCEPT
};

class RangeAssumptionStmt : public Stmt {
 public:
  Stmt *input;
  Stmt *base;
  int low, high;

  RangeAssumptionStmt(Stmt *input, Stmt *base, int low, int high)
      : input(input), base(base), low(low), high(high) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(ret_type, input, base, low, high);
  DEFINE_ACCEPT
};

class GlobalLoadStmt : public Stmt {
 public:
  Stmt *ptr;

  GlobalLoadStmt(Stmt *ptr) : ptr(ptr) {
    TI_STMT_REG_FIELDS;
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, ptr);
  DEFINE_ACCEPT;
};

class GlobalStoreStmt : public Stmt {
 public:
  Stmt *ptr, *data;

  GlobalStoreStmt(Stmt *ptr, Stmt *data) : ptr(ptr), data(data) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(ret_type, ptr, data);
  DEFINE_ACCEPT;
};

struct LocalAddress {
  Stmt *var;
  int offset;

  LocalAddress(Stmt *var, int offset) : var(var), offset(offset) {
    TI_ASSERT(var->is<AllocaStmt>());
  }
};

template <typename T>
std::string to_string(const T &);

class LocalLoadStmt : public Stmt {
 public:
  LaneAttribute<LocalAddress> ptr;

  LocalLoadStmt(const LaneAttribute<LocalAddress> &ptr) : ptr(ptr) {
    TI_STMT_REG_FIELDS;
  }

  void rebuild_operands() override;
  bool same_source() const;
  bool has_source(Stmt *alloca) const;

  bool integral_operands() const override {
    return false;
  }

  Stmt *previous_store_or_alloca_in_block();

  virtual bool has_global_side_effect() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, ptr);
  DEFINE_ACCEPT;
};

class LocalStoreStmt : public Stmt {
 public:
  Stmt *ptr;
  Stmt *data;

  // LaneAttribute<Stmt *> data;

  LocalStoreStmt(Stmt *ptr, Stmt *data) : ptr(ptr), data(data) {
    TI_ASSERT(ptr->is<AllocaStmt>());
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(ret_type, ptr, data);
  DEFINE_ACCEPT;
};

class IfStmt : public Stmt {
 public:
  Stmt *cond;
  Stmt *true_mask, *false_mask;
  std::unique_ptr<Block> true_statements, false_statements;

  IfStmt(Stmt *cond) : cond(cond), true_mask(nullptr), false_mask(nullptr) {
    TI_STMT_REG_FIELDS;
  }

  bool is_container_statement() const override {
    return true;
  }

  TI_STMT_DEF_FIELDS(cond, true_mask, false_mask);
  DEFINE_ACCEPT
};

class PrintStmt : public Stmt {
 public:
  Stmt *stmt;
  std::string str;

  PrintStmt(Stmt *stmt, const std::string &str) : stmt(stmt), str(str) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(ret_type, stmt, str);
  DEFINE_ACCEPT
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
    TI_STMT_REG_FIELDS;
  }

  void repeat(int factor) override {
    Stmt::repeat(factor);
    val.repeat(factor);
  }

  virtual bool has_global_side_effect() const override {
    return false;
  }

  std::unique_ptr<ConstStmt> copy();

  TI_STMT_DEF_FIELDS(ret_type, val);
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
    TI_STMT_REG_FIELDS;
  }

  bool is_container_statement() const override {
    return true;
  }

  void reverse() {
    reversed = !reversed;
  }

  TI_STMT_DEF_FIELDS(loop_var,
                     begin,
                     end,
                     reversed,
                     vectorize,
                     parallelize,
                     block_dim,
                     strictly_serialized);
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
    TI_STMT_REG_FIELDS;
  }

  bool is_container_statement() const override {
    return true;
  }

  TI_STMT_DEF_FIELDS(loop_vars, snode, vectorize, parallelize, block_dim);
  DEFINE_ACCEPT
};

class FuncBodyStmt : public Stmt {
 public:
  std::string funcid;
  std::unique_ptr<Block> body;

  FuncBodyStmt(const std::string &funcid, std::unique_ptr<Block> &&body)
      : funcid(funcid), body(std::move(body)) {
    TI_STMT_REG_FIELDS;
  }

  bool is_container_statement() const override {
    return true;
  }

  TI_STMT_DEF_FIELDS(funcid);
  DEFINE_ACCEPT
};

class FuncCallStmt : public Stmt {
 public:
  std::string funcid;

  FuncCallStmt(const std::string &funcid) : funcid(funcid) {
    TI_STMT_REG_FIELDS;
  }

  bool is_container_statement() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(ret_type, funcid);
  DEFINE_ACCEPT
};

class KernelReturnStmt : public Stmt {
 public:
  Stmt *value;

  KernelReturnStmt(Stmt *value) : value(value) {
    TI_STMT_REG_FIELDS;
  }

  bool is_container_statement() const override {
    return false;
  }

  TI_STMT_DEF_FIELDS(value);
  DEFINE_ACCEPT
};

class WhileStmt : public Stmt {
 public:
  Stmt *mask;
  std::unique_ptr<Block> body;

  WhileStmt(std::unique_ptr<Block> &&body)
      : mask(nullptr), body(std::move(body)) {
    TI_STMT_REG_FIELDS;
  }

  bool is_container_statement() const override {
    return true;
  }

  TI_STMT_DEF_FIELDS(mask);
  DEFINE_ACCEPT
};

void Print_(const Expr &a, const std::string &str);

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

Expr Var(const Expr &x);

class VectorElement {
 public:
  Stmt *stmt;
  int index;

  VectorElement() : stmt(nullptr), index(0) {
  }

  VectorElement(Stmt *stmt, int index) : stmt(stmt), index(index) {
  }
};

template <typename T>
inline void StmtFieldManager::operator()(const char *key, T &&value) {
  using decay_T = typename std::decay<T>::type;
  if constexpr (is_specialization<decay_T, std::vector>::value ||
                is_specialization<decay_T, LaneAttribute>::value) {
    stmt->field_manager.fields.emplace_back(
        std::make_unique<StmtFieldNumeric<std::size_t>>(value.size()));
    for (int i = 0; i < (int)value.size(); i++) {
      (*this)("__element", value[i]);
    }
  } else if constexpr (std::is_same<decay_T, Stmt *>::value) {
    stmt->register_operand(const_cast<Stmt *&>(value));
  } else if constexpr (std::is_same<decay_T, LocalAddress>::value) {
    stmt->register_operand(const_cast<Stmt *&>(value.var));
    stmt->field_manager.fields.emplace_back(
        std::make_unique<StmtFieldNumeric<int>>(value.offset));
  } else if constexpr (std::is_same<decay_T, VectorElement>::value) {
    stmt->register_operand(const_cast<Stmt *&>(value.stmt));
    stmt->field_manager.fields.emplace_back(
        std::make_unique<StmtFieldNumeric<int>>(value.index));
  } else if constexpr (std::is_same<decay_T, SNode *>::value) {
    stmt->field_manager.fields.emplace_back(
        std::make_unique<StmtFieldSNode>(value));
  } else {
    stmt->field_manager.fields.emplace_back(
        std::make_unique<StmtFieldNumeric<T>>(value));
  }
}

TLANG_NAMESPACE_END

#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
