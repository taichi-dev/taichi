// Intermediate representation system

#pragma once

#include <atomic>
#include <unordered_set>
#include <unordered_map>
#include <variant>
#include <tuple>

#include "taichi/common/core.h"
#include "taichi/util/bit.h"
#include "taichi/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/program/compile_config.h"
#include "taichi/llvm/llvm_fwd.h"
#include "taichi/util/short_name.h"

TLANG_NAMESPACE_BEGIN

class IRBuilder;
class IRNode;
class Block;
class Stmt;
using pStmt = std::unique_ptr<Stmt>;

class SNode;
class ScratchPads;
using ScratchPadOptions = std::vector<std::pair<int, SNode *>>;

#define PER_STATEMENT(x) class x;
#include "taichi/inc/statements.inc.h"
#undef PER_STATEMENT

IRBuilder &current_ast_builder();

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

  virtual ~IRVisitor() = default;

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

struct CompileConfig;

class IRNode {
 public:
  Kernel *kernel;

  virtual void accept(IRVisitor *visitor) {
    TI_NOT_IMPLEMENTED
  }

  // * For a Stmt, this returns its enclosing Block
  // * For a Block, this returns its enclosing Stmt
  virtual IRNode *get_parent() const = 0;

  IRNode *get_ir_root();
  Kernel *get_kernel() const;

  virtual ~IRNode() = default;

  CompileConfig &get_config() const;

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
  const T *as() const {
    TI_ASSERT(is<T>());
    return dynamic_cast<const T *>(this);
  }

  template <typename T>
  T *cast() {
    return dynamic_cast<T *>(this);
  }

  template <typename T>
  const T *cast() const {
    return dynamic_cast<const T *>(this);
  }

  std::unique_ptr<IRNode> clone();
};

#define TI_DEFINE_ACCEPT                     \
  void accept(IRVisitor *visitor) override { \
    visitor->visit(this);                    \
  }

#define TI_DEFINE_CLONE                                             \
  std::unique_ptr<Stmt> clone() const override {                    \
    auto new_stmt =                                                 \
        std::make_unique<std::decay<decltype(*this)>::type>(*this); \
    new_stmt->mark_fields_registered();                             \
    new_stmt->io(new_stmt->field_manager);                          \
    return new_stmt;                                                \
  }

#define TI_DEFINE_ACCEPT_AND_CLONE \
  TI_DEFINE_ACCEPT                 \
  TI_DEFINE_CLONE

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
  std::variant<T *, T> value;

 public:
  explicit StmtFieldNumeric(T *value) : value(value) {
  }

  explicit StmtFieldNumeric(T value) : value(value) {
  }

  bool equal(const StmtField *other_generic) const override {
    if (auto other = dynamic_cast<const StmtFieldNumeric *>(other_generic)) {
      if (std::holds_alternative<T *>(other->value) &&
          std::holds_alternative<T *>(value)) {
        return *(std::get<T *>(other->value)) == *(std::get<T *>(value));
      } else if (std::holds_alternative<T *>(other->value) ||
                 std::holds_alternative<T *>(value)) {
        TI_ERROR(
            "Inconsistent StmtField value types: a pointer value is compared "
            "to a non-pointer value.");
        return false;
      } else {
        return std::get<T>(other->value) == std::get<T>(value);
      }
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
 protected:
  std::vector<Stmt **> operands;

 public:
  StmtFieldManager field_manager;
  static std::atomic<int> instance_id_counter;
  int instance_id;
  int id;
  Block *parent;
  bool erased;
  bool fields_registered;
  std::string tb;
  DataType ret_type;

  Stmt();
  Stmt(const Stmt &stmt);

  int width() const {
    return ret_type->vector_width();
  }

  virtual bool is_container_statement() const {
    return false;
  }

  DataType &element_type() {
    return ret_type;
  }

  std::string ret_data_type_name() const {
    return ret_type->to_string();
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

  void set_operand(int i, Stmt *stmt);
  void register_operand(Stmt *&stmt);
  int locate_operand(Stmt **stmt);
  void mark_fields_registered();

  bool has_operand(Stmt *stmt) const;

  void replace_with(Stmt *new_stmt);
  void replace_with(VecStatement &&new_statements, bool replace_usages = true);
  virtual void replace_operand_with(Stmt *old_stmt, Stmt *new_stmt);

  IRNode *get_parent() const override;

  virtual void repeat(int factor) {
    TI_ASSERT(factor == 1);
    // ret_type.width *= factor;
  }

  // returns the inserted stmt
  Stmt *insert_before_me(std::unique_ptr<Stmt> &&new_stmt);

  // returns the inserted stmt
  Stmt *insert_after_me(std::unique_ptr<Stmt> &&new_stmt);

  virtual bool has_global_side_effect() const {
    return true;
  }

  virtual bool dead_instruction_eliminable() const {
    return !has_global_side_effect();
  }

  virtual bool common_statement_eliminable() const {
    return !has_global_side_effect();
  }

  template <typename T, typename... Args>
  static std::unique_ptr<T> make_typed(Args &&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  static pStmt make(Args &&... args) {
    return make_typed<T>(std::forward<Args>(args)...);
  }

  void infer_type();

  void set_tb(const std::string &tb) {
    this->tb = tb;
  }

  std::string type();

  virtual std::unique_ptr<Stmt> clone() const {
    TI_NOT_IMPLEMENTED
  }

  virtual ~Stmt() override = default;
};

class Block : public IRNode {
 public:
  Stmt *parent_stmt;
  std::vector<std::unique_ptr<Stmt>> statements, trash_bin;
  Stmt *mask_var;
  std::vector<SNode *> stop_gradients;

  // Only used in frontend. Stores LoopIndexStmt or BinaryOpStmt for loop
  // variables, and AllocaStmt for other variables.
  std::map<Identifier, Stmt *> local_var_to_stmt;

  Block() {
    mask_var = nullptr;
    parent_stmt = nullptr;
    kernel = nullptr;
  }

  Block *parent_block() const;

  bool has_container_statements();
  int locate(Stmt *stmt);
  void erase(int location);
  void erase(Stmt *stmt);
  std::unique_ptr<Stmt> extract(int location);
  std::unique_ptr<Stmt> extract(Stmt *stmt);

  // Returns stmt.get()
  Stmt *insert(std::unique_ptr<Stmt> &&stmt, int location = -1);

  // Returns stmt.back().get() or nullptr if stmt is empty
  Stmt *insert(VecStatement &&stmt, int location = -1);

  void replace_statements_in_range(int start, int end, VecStatement &&stmts);
  void set_statements(VecStatement &&stmts);
  void replace_with(Stmt *old_statement,
                    std::unique_ptr<Stmt> &&new_statement,
                    bool replace_usages = true);
  void insert_before(Stmt *old_statement, VecStatement &&new_statements);
  void insert_after(Stmt *old_statement, VecStatement &&new_statements);
  void replace_with(Stmt *old_statement,
                    VecStatement &&new_statements,
                    bool replace_usages = true);
  Stmt *lookup_var(const Identifier &ident) const;
  Stmt *mask();
  IRNode *get_parent() const override;

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

  std::size_t size() const {
    return statements.size();
  }

  pStmt &operator[](int i) {
    return statements[i];
  }

  std::unique_ptr<Block> clone() const;

  TI_DEFINE_ACCEPT
};

class DelayedIRModifier {
 private:
  std::vector<std::pair<Stmt *, VecStatement>> to_insert_before;
  std::vector<std::pair<Stmt *, VecStatement>> to_insert_after;
  std::vector<std::tuple<Stmt *, VecStatement, bool>> to_replace_with;
  std::vector<Stmt *> to_erase;

 public:
  ~DelayedIRModifier();
  void erase(Stmt *stmt);
  void insert_before(Stmt *old_statement, std::unique_ptr<Stmt> new_statement);
  void insert_before(Stmt *old_statement, VecStatement &&new_statements);
  void insert_after(Stmt *old_statement, std::unique_ptr<Stmt> new_statement);
  void insert_after(Stmt *old_statement, VecStatement &&new_statements);
  void replace_with(Stmt *stmt,
                    VecStatement &&new_statements,
                    bool replace_usages = true);
  bool modify_ir();
};

struct LocalAddress {
  Stmt *var;
  int offset;

  LocalAddress(Stmt *var, int offset);
};

template <typename T>
std::string to_string(const T &);

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

inline void BlockDim(int v) {
  TI_ASSERT(bit::is_power_of_two(v));
  dec.block_dim = v;
}

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
  } else if constexpr (std::is_same<decay_T,
                                    std::variant<Stmt *, std::string>>::value) {
    if (std::holds_alternative<std::string>(value)) {
      stmt->field_manager.fields.emplace_back(
          std::make_unique<StmtFieldNumeric<std::string>>(
              std::get<std::string>(value)));
    } else {
      (*this)("__element", std::get<Stmt *>(value));
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
        std::make_unique<StmtFieldNumeric<std::remove_reference_t<T>>>(&value));
  }
}

TLANG_NAMESPACE_END
