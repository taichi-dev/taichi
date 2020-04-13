// Intermediate representations

#include "taichi/ir/ir.h"

#include <numeric>
#include <thread>
#include <unordered_map>

#include "taichi/ir/frontend.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

#define TI_EXPRESSION_IMPLEMENTATION
#include "expression.h"

IRBuilder &current_ast_builder() {
  return context->builder();
}

std::string VectorType::pointer_suffix() const {
  if (is_pointer()) {
    return "*";
  } else {
    return "";
  }
}

std::string VectorType::element_type_name() const {
  return fmt::format("{}{}", data_type_short_name(data_type),
                     pointer_suffix());
}

std::string VectorType::str() const {
  auto ename = element_type_name();
  return fmt::format("{:4}x{}", ename, width);
}

void DecoratorRecorder::reset() {
  vectorize = -1;
  parallelize = 0;
  uniform = false;
  scratch_opt.clear();
  block_dim = 0;
  strictly_serialized = false;
}

Block *IRBuilder::current_block() {
  if (stack.empty())
    return nullptr;
  else
    return stack.back();
}

Stmt *IRBuilder::get_last_stmt() {
  return stack.back()->back();
}

void IRBuilder::insert(std::unique_ptr<Stmt> &&stmt, int location) {
  TI_ASSERT(!stack.empty());
  stack.back()->insert(std::move(stmt), location);
}

void IRBuilder::stop_gradient(SNode *snode) {
  TI_ASSERT(!stack.empty());
  stack.back()->stop_gradients.push_back(snode);
}

std::unique_ptr<IRBuilder::ScopeGuard> IRBuilder::create_scope(
    std::unique_ptr<Block> &list) {
  TI_ASSERT(list == nullptr);
  list = std::make_unique<Block>();
  if (!stack.empty()) {
    list->parent = stack.back();
  }
  return std::make_unique<ScopeGuard>(this, list.get());
}

int Identifier::id_counter = 0;
std::string Identifier::raw_name() const {
  if (name_.empty())
    return fmt::format("tmp{}", id);
  else
    return name_;
}

Stmt *VecStatement::push_back(pStmt &&stmt) {
  auto ret = stmt.get();
  stmts.push_back(std::move(stmt));
  return ret;
}

class StatementTypeNameVisitor : public IRVisitor {
 public:
  std::string type_name;
  StatementTypeNameVisitor() {
  }

#define PER_STATEMENT(x)         \
  void visit(x *stmt) override { \
    type_name = #x;              \
  }
#include "taichi/inc/statements.inc.h"

#undef PER_STATEMENT
};

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

inline Expr smart_load(const Expr &var) {
  return load_if_ptr(ptr_if_global(var));
}

int StmtFieldSNode::get_snode_id(taichi::lang::SNode *snode) {
  if (snode == nullptr)
    return -1;
  return snode->id;
}

bool StmtFieldSNode::equal(const StmtField *other_generic) const {
  if (auto other = dynamic_cast<const StmtFieldSNode *>(other_generic)) {
    return get_snode_id(snode) == get_snode_id(other->snode);
  } else {
    // Different types
    return false;
  }
}

bool StmtFieldManager::equal(StmtFieldManager &other) const {
  if (fields.size() != other.fields.size()) {
    return false;
  }
  auto num_fields = fields.size();
  for (std::size_t i = 0; i < num_fields; i++) {
    if (!fields[i]->equal(other.fields[i].get())) {
      return false;
    }
  }
  return true;
}

std::atomic<int> Stmt::instance_id_counter(0);

Stmt::Stmt() : field_manager(this), fields_registered(false) {
  parent = nullptr;
  instance_id = instance_id_counter++;
  id = instance_id;
  operand_bitmap = 0;
  erased = false;
  is_ptr = false;
}

Stmt *Stmt::insert_before_me(std::unique_ptr<Stmt> &&new_stmt) {
  auto ret = new_stmt.get();
  TI_ASSERT(parent);
  auto &stmts = parent->statements;
  int loc = -1;
  for (int i = 0; i < (int)stmts.size(); i++) {
    if (stmts[i].get() == this) {
      loc = i;
      break;
    }
  }
  TI_ASSERT(loc != -1);
  new_stmt->parent = parent;
  stmts.insert(stmts.begin() + loc, std::move(new_stmt));
  return ret;
}

Stmt *Stmt::insert_after_me(std::unique_ptr<Stmt> &&new_stmt) {
  auto ret = new_stmt.get();
  TI_ASSERT(parent);
  auto &stmts = parent->statements;
  int loc = -1;
  for (int i = 0; i < (int)stmts.size(); i++) {
    if (stmts[i].get() == this) {
      loc = i;
      break;
    }
  }
  TI_ASSERT(loc != -1);
  new_stmt->parent = parent;
  stmts.insert(stmts.begin() + loc + 1, std::move(new_stmt));
  return ret;
}

void Stmt::replace_with(Stmt *new_stmt) {
  auto root = get_ir_root();
  irpass::replace_all_usages_with(root, this, new_stmt);
  // Note: the current structure should have been destroyed now..
}

void Stmt::replace_with(VecStatement &&new_statements, bool replace_usages) {
  parent->replace_with(this, std::move(new_statements), replace_usages);
}

void Stmt::replace_operand_with(Stmt *old_stmt, Stmt *new_stmt) {
  operand_bitmap = 0;
  int n_op = num_operands();
  for (int i = 0; i < n_op; i++) {
    if (operand(i) == old_stmt) {
      *operands[i] = new_stmt;
    }
    operand_bitmap |= operand_hash(operand(i));
  }
  rebuild_operand_bitmap();
}

std::string Stmt::type_hint() const {
  if (ret_type.data_type == DataType::unknown)
    return "";
  else
    return fmt::format("<{}>{}", ret_type.str(), is_ptr ? "ptr " : " ");
}

std::string Stmt::type() {
  StatementTypeNameVisitor v;
  this->accept(&v);
  return v.type_name;
}

IRNode *Stmt::get_ir_root() {
  auto block = parent;
  while (block->parent)
    block = block->parent;
  return dynamic_cast<IRNode *>(block);
}

std::vector<Stmt *> Stmt::get_operands() const {
  std::vector<Stmt *> ret;
  for (int i = 0; i < num_operands(); i++) {
    ret.push_back(*operands[i]);
  }
  return ret;
}

void Stmt::set_operand(int i, Stmt *stmt) {
  *operands[i] = stmt;
  rebuild_operand_bitmap();
}

void Stmt::register_operand(Stmt *&stmt) {
  operands.push_back(&stmt);
  rebuild_operand_bitmap();
}

void Stmt::mark_fields_registered() {
  TI_ASSERT(!fields_registered);
  fields_registered = true;
}

std::string Expression::get_attribute(
    const std::string &key) const {
  if (auto it = attributes.find(key); it == attributes.end()) {
    TI_ERROR("Attribute {} not found.", key);
  } else {
    return it->second;
  }
}

ExprGroup ExprGroup::loaded() const {
  auto indices_loaded = *this;
  for (int i = 0; i < (int)this->size(); i++)
    indices_loaded[i].set(load_if_ptr(indices_loaded[i]));
  return indices_loaded;
}

std::string ExprGroup::serialize() const {
  std::string ret;
  for (int i = 0; i < (int)exprs.size(); i++) {
    ret += exprs[i].serialize();
    if (i + 1 < (int)exprs.size()) {
      ret += ", ";
    }
  }
  return ret;
}

UnaryOpStmt::UnaryOpStmt(taichi::lang::UnaryOpType op_type,
                         taichi::lang::Stmt *operand)
    : op_type(op_type), operand(operand) {
  TI_ASSERT(!operand->is<AllocaStmt>());
  cast_type = DataType::unknown;
  cast_by_value = true;
  TI_STMT_REG_FIELDS;
}

bool UnaryOpStmt::same_operation(UnaryOpStmt *o) const {
  if (op_type == o->op_type) {
    if (op_type == UnaryOpType::cast) {
      return cast_type == o->cast_type;
    } else {
      return true;
    }
  }
  return false;
}

std::string UnaryOpExpression::serialize() {
  if (type == UnaryOpType::cast) {
    std::string reint = cast_by_value ? "" : "reinterpret_";
    return fmt::format("({}{}<{}> {})", reint, unary_op_type_name(type),
                       data_type_name(cast_type), operand->serialize());
  } else {
    return fmt::format("({} {})", unary_op_type_name(type),
                       operand->serialize());
  }
}

void UnaryOpExpression::flatten(VecStatement &ret) {
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

ExternalPtrStmt::ExternalPtrStmt(
    const taichi::lang::LaneAttribute<taichi::lang::Stmt *> &base_ptrs,
    const std::vector<Stmt *> &indices)
    : base_ptrs(base_ptrs), indices(indices) {
  DataType dt = DataType::f32;
  for (int i = 0; i < (int)base_ptrs.size(); i++) {
    TI_ASSERT(base_ptrs[i] != nullptr);
    TI_ASSERT(base_ptrs[i]->is<ArgLoadStmt>());
  }
  width() = base_ptrs.size();
  element_type() = dt;
  TI_STMT_REG_FIELDS;
}

GlobalPtrStmt::GlobalPtrStmt(
    const taichi::lang::LaneAttribute<taichi::lang::SNode *> &snodes,
    const std::vector<Stmt *> &indices,
    bool activate)
    : snodes(snodes), indices(indices), activate(activate) {
  for (int i = 0; i < (int)snodes.size(); i++) {
    TI_ASSERT(snodes[i] != nullptr);
    TI_ASSERT(snodes[0]->dt == snodes[i]->dt);
  }
  width() = snodes.size();
  element_type() = snodes[0]->dt;
  TI_STMT_REG_FIELDS;
}

std::string GlobalPtrExpression::serialize() {
  std::string s = fmt::format("{}[", var.serialize());
  for (int i = 0; i < (int)indices.size(); i++) {
    s += indices.exprs[i]->serialize();
    if (i + 1 < (int)indices.size())
      s += ", ";
  }
  s += "]";
  return s;
}

void GlobalPtrExpression::flatten(VecStatement &ret) {
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

GetChStmt::GetChStmt(taichi::lang::Stmt *input_ptr, int chid)
    : input_ptr(input_ptr), chid(chid) {
  TI_ASSERT(input_ptr->is<SNodeLookupStmt>());
  input_snode = input_ptr->as<SNodeLookupStmt>()->snode;
  output_snode = input_snode->ch[chid].get();
  TI_STMT_REG_FIELDS;
}

DecoratorRecorder dec;

FrontendContext::FrontendContext() {
  root_node = std::make_unique<Block>();
  current_builder = std::make_unique<IRBuilder>(root_node.get());
}

Expr::Expr(int32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(int64 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(float32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(float64 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(const Identifier &id) : Expr() {
  expr = std::make_shared<IdExpression>(id);
}

Expr Expr::eval() const {
  TI_ASSERT(expr != nullptr);
  if (is<EvalExpression>()) {
    return *this;
  }
  auto eval_stmt = Stmt::make<FrontendEvalStmt>(*this);
  auto eval_expr = Expr::make<EvalExpression>(eval_stmt.get());
  eval_stmt->as<FrontendEvalStmt>()->eval_expr.set(eval_expr);
  // needed in lower_ast to replace the statement itself with the
  // lowered statement
  current_ast_builder().insert(std::move(eval_stmt));
  return eval_expr;
}

void Expr::operator+=(const Expr &o) {
  if (this->atomic) {
    current_ast_builder().insert(Stmt::make<FrontendAtomicStmt>(
        AtomicOpType::add, ptr_if_global(*this), load_if_ptr(o)));
  } else {
    (*this) = (*this) + o;
  }
}
void Expr::operator-=(const Expr &o) {
  if (this->atomic) {
    current_ast_builder().insert(Stmt::make<FrontendAtomicStmt>(
        AtomicOpType::add, *this, -load_if_ptr(o)));
  } else {
    (*this) = (*this) - o;
  }
}
void Expr::operator*=(const Expr &o) {
  TI_ASSERT(!this->atomic);
  (*this) = (*this) * load_if_ptr(o);
}
void Expr::operator/=(const Expr &o) {
  TI_ASSERT(!this->atomic);
  (*this) = (*this) / load_if_ptr(o);
}

FrontendForStmt::FrontendForStmt(const Expr &loop_var,
                                 const Expr &begin,
                                 const Expr &end)
    : begin(begin), end(end) {
  vectorize = dec.vectorize;
  parallelize = dec.parallelize;
  strictly_serialized = dec.strictly_serialized;
  block_dim = dec.block_dim;
  auto cfg = get_current_program().config;
  if (cfg.arch == Arch::cuda) {
    vectorize = 1;
    parallelize = 1;
  } else {
    if (block_dim == 0)
      block_dim = cfg.default_cpu_block_dim;
    if (parallelize == 0)
      parallelize = std::thread::hardware_concurrency();
  }
  scratch_opt = dec.scratch_opt;
  dec.reset();
  if (vectorize == -1)
    vectorize = 1;
  loop_var_id.resize(1);
  loop_var_id[0] = loop_var.cast<IdExpression>()->id;
}

FrontendForStmt::FrontendForStmt(const ExprGroup &loop_var,
                                 const Expr &global_var)
    : global_var(global_var) {
  vectorize = dec.vectorize;
  parallelize = dec.parallelize;
  strictly_serialized = dec.strictly_serialized;
  block_dim = dec.block_dim;
  auto cfg = get_current_program().config;
  if (cfg.arch == Arch::cuda) {
    vectorize = 1;
    parallelize = 1;
    TI_ASSERT(block_dim <= taichi_max_gpu_block_dim);
  } else {
    // cpu
    if (block_dim == 0)
      block_dim = cfg.default_cpu_block_dim;
    if (parallelize == 0)
      parallelize = std::thread::hardware_concurrency();
  }
  scratch_opt = dec.scratch_opt;
  dec.reset();
  if (vectorize == -1)
    vectorize = 1;

  loop_var_id.resize(loop_var.size());
  for (int i = 0; i < (int)loop_var.size(); i++) {
    loop_var_id[i] = loop_var[i].cast<IdExpression>()->id;
  }
}


FrontendAssignStmt::FrontendAssignStmt(const Expr &lhs, const Expr &rhs)
    : lhs(lhs), rhs(rhs) {
  TI_ASSERT(lhs->is_lvalue());
}

FrontendAtomicStmt::FrontendAtomicStmt(AtomicOpType op_type,
                                       const Expr &dest,
                                       const Expr &val)
    : op_type(op_type), dest(dest), val(val) {
}

IRNode *FrontendContext::root() {
  return static_cast<IRNode *>(root_node.get());
}

std::unique_ptr<FrontendContext> context;

Block *current_block = nullptr;

Expr Var(const Expr &x) {
  auto var = Expr(std::make_shared<IdExpression>());
  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id, DataType::unknown));
  var = x;
  return var;
}

void Print_(const Expr &a, const std::string &str) {
  current_ast_builder().insert(std::make_unique<FrontendPrintStmt>(a, str));
}

template <>
std::string to_string(const LaneAttribute<LocalAddress> &ptr) {
  std::string ret = " [";
  for (int i = 0; i < (int)ptr.size(); i++) {
    ret += fmt::format("{}[{}]", ptr[i].var->name(), ptr[i].offset);
    if (i + 1 < (int)ptr.size())
      ret += ", ";
  }
  ret += "]";
  return ret;
}

Stmt *LocalLoadStmt::previous_store_or_alloca_in_block() {
  int position = parent->locate(this);
  // TI_ASSERT(width() == 1);
  // TI_ASSERT(this->ptr[0].offset == 0);
  for (int i = position - 1; i >= 0; i--) {
    if (parent->statements[i]->is<LocalStoreStmt>()) {
      auto store = parent->statements[i]->as<LocalStoreStmt>();
      // TI_ASSERT(store->width() == 1);
      if (store->ptr == this->ptr[0].var) {
        // found
        return store;
      }
    } else if (parent->statements[i]->is<AllocaStmt>()) {
      auto alloca = parent->statements[i]->as<AllocaStmt>();
      // TI_ASSERT(alloca->width() == 1);
      if (alloca == this->ptr[0].var) {
        return alloca;
      }
    }
  }
  return nullptr;
}

void Block::erase(int location) {
  statements[location]->erased = true;
  trash_bin.push_back(std::move(statements[location]));  // do not delete the
  // stmt, otherwise print_ir will not function properly
  statements.erase(statements.begin() + location);
}

void Block::erase(Stmt *stmt) {
  for (int i = 0; i < (int)statements.size(); i++) {
    if (statements[i].get() == stmt) {
      erase(i);
      break;
    }
  }
}

std::unique_ptr<Stmt> Block::extract(int location) {
  auto stmt = std::move(statements[location]);
  statements.erase(statements.begin() + location);
  return stmt;
}

std::unique_ptr<Stmt> Block::extract(Stmt *stmt) {
  for (int i = 0; i < (int)statements.size(); i++) {
    if (statements[i].get() == stmt) {
      return extract(i);
    }
  }
  TI_ERROR("stmt not found");
}

void Block::insert(std::unique_ptr<Stmt> &&stmt, int location) {
  stmt->parent = this;
  if (location == -1) {
    statements.push_back(std::move(stmt));
  } else {
    statements.insert(statements.begin() + location, std::move(stmt));
  }
}

void Block::insert(VecStatement &&stmt, int location) {
  if (location == -1) {
    location = (int)statements.size() - 1;
  }
  for (int i = 0; i < stmt.size(); i++) {
    insert(std::move(stmt[i]), location + i);
  }
}

void Block::replace_statements_in_range(int start,
                                        int end,
                                        VecStatement &&stmts) {
  TI_ASSERT(start <= end);
  for (int i = 0; i < end - start; i++) {
    erase(start);
  }

  for (int i = 0; i < (int)stmts.size(); i++) {
    insert(std::move(stmts[i]), start + i);
  }
}

void Block::replace_with(Stmt *old_statement,
                         std::unique_ptr<Stmt> &&new_statement) {
  VecStatement vec;
  vec.push_back(std::move(new_statement));
  replace_with(old_statement, std::move(vec));
}

Stmt *Block::lookup_var(const taichi::lang::Ident &ident) const {
  auto ptr = local_var_alloca.find(ident);
  if (ptr != local_var_alloca.end()) {
    return ptr->second;
  } else {
    if (parent) {
      return parent->lookup_var(ident);
    } else {
      return nullptr;
    }
  }
}

Stmt *Block::mask() {
  if (mask_var)
    return mask_var;
  else if (parent == nullptr) {
    return nullptr;
  } else {
    return parent->mask();
  }
}

For::For(const Expr &s, const Expr &e, const std::function<void(Expr)> &func) {
  auto i = Expr(std::make_shared<IdExpression>());
  auto stmt_unique = std::make_unique<FrontendForStmt>(i, s, e);
  auto stmt = stmt_unique.get();
  current_ast_builder().insert(std::move(stmt_unique));
  auto _ = current_ast_builder().create_scope(stmt->body);
  func(i);
}

OffloadedStmt::OffloadedStmt(OffloadedStmt::TaskType task_type)
    : OffloadedStmt(task_type, nullptr) {
}

OffloadedStmt::OffloadedStmt(OffloadedStmt::TaskType task_type, SNode *snode)
    : task_type(task_type), snode(snode) {
  num_cpu_threads = 1;
  const_begin = false;
  const_end = false;
  begin_value = 0;
  end_value = 0;
  step = 0;
  block_dim = 0;
  reversed = false;
  device = get_current_program().config.arch;
  if (task_type != TaskType::listgen) {
    body = std::make_unique<Block>();
  }
  TI_STMT_REG_FIELDS;
}

std::string OffloadedStmt::task_name() const {
  if (task_type == TaskType::serial) {
    return "serial";
  } else if (task_type == TaskType::range_for) {
    return "range_for";
  } else if (task_type == TaskType::struct_for) {
    return "struct_for";
  } else if (task_type == TaskType::clear_list) {
    TI_ASSERT(snode);
    return fmt::format("clear_list_{}", snode->get_node_type_name_hinted());
  } else if (task_type == TaskType::listgen) {
    TI_ASSERT(snode);
    return fmt::format("listgen_{}", snode->get_node_type_name_hinted());
  } else if (task_type == TaskType::gc) {
    TI_ASSERT(snode);
    return fmt::format("gc_{}", snode->name);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

// static
std::string OffloadedStmt::task_type_name(TaskType tt) {
#define REGISTER_NAME(x) \
  { TaskType::x, #x }
  const static std::unordered_map<TaskType, std::string> m = {
      REGISTER_NAME(serial),     REGISTER_NAME(range_for),
      REGISTER_NAME(struct_for), REGISTER_NAME(clear_list),
      REGISTER_NAME(listgen),    REGISTER_NAME(gc),
  };
#undef REGISTER_NAME
  return m.find(tt)->second;
}

bool ContinueStmt::as_return() const {
  TI_ASSERT(scope != nullptr);
  if (auto *offl = scope->cast<OffloadedStmt>(); offl) {
    TI_ASSERT(offl->task_type == OffloadedStmt::TaskType::range_for ||
              offl->task_type == OffloadedStmt::TaskType::struct_for);
    return true;
  }
  return false;
}

TLANG_NAMESPACE_END
