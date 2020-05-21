// Intermediate representations

#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"

#include <numeric>
#include <thread>
#include <unordered_map>

#include "taichi/ir/frontend.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

#define TI_EXPRESSION_IMPLEMENTATION
#include "expression_ops.h"

IRBuilder &current_ast_builder() {
  return context->builder();
}

bool maybe_same_address(Stmt *var1, Stmt *var2) {
  // Return true when two statements might be the same address;
  // false when two statements cannot be the same address.

  // If both stmts are allocas, they have the same address iff var1 == var2.
  // If only one of them is an alloca, they can never share the same address.
  if (var1 == var2)
    return true;
  if (var1->is<AllocaStmt>() || var2->is<AllocaStmt>())
    return false;

  // If both statements are global temps, they have the same address iff they
  // have the same offset. If only one of them is a global temp, they can never
  // share the same address.
  if (var1->is<GlobalTemporaryStmt>() || var2->is<GlobalTemporaryStmt>()) {
    if (!var1->is<GlobalTemporaryStmt>() || !var2->is<GlobalTemporaryStmt>())
      return false;
    return var1->as<GlobalTemporaryStmt>()->offset ==
           var2->as<GlobalTemporaryStmt>()->offset;
  }

  // If both statements are GlobalPtrStmts or GetChStmts, we can check by
  // SNode::id.
  TI_ASSERT(var1->width() == 1);
  TI_ASSERT(var2->width() == 1);
  auto get_snode_id = [](Stmt *s) {
    if (auto ptr = s->cast<GlobalPtrStmt>())
      return ptr->snodes[0]->id;
    else if (auto get_child = s->cast<GetChStmt>())
      return get_child->output_snode->id;
    else
      return -1;
  };
  int snode1 = get_snode_id(var1);
  int snode2 = get_snode_id(var2);
  if (snode1 != -1 && snode2 != -1 && snode1 != snode2)
    return false;

  // GlobalPtrStmts with guaranteed different indices cannot share the same
  // address.
  if (var1->is<GlobalPtrStmt>() && var2->is<GlobalPtrStmt>()) {
    auto ptr1 = var1->as<GlobalPtrStmt>();
    auto ptr2 = var2->as<GlobalPtrStmt>();
    for (int i = 0; i < (int)ptr1->indices.size(); i++) {
      if (!irpass::analysis::same_statements(ptr1->indices[i],
                                             ptr2->indices[i])) {
        if (ptr1->indices[i]->is<ConstStmt>() &&
            ptr2->indices[i]->is<ConstStmt>()) {
          // different constants
          return false;
        }
      }
    }
    return true;
  }

  // In other cases (probably after lower_access), we don't know if the two
  // statements share the same address.
  return true;
}

std::string VectorType::pointer_suffix() const {
  if (is_pointer()) {
    return "*";
  } else {
    return "";
  }
}

std::string VectorType::element_type_name() const {
  return fmt::format("{}{}", data_type_short_name(data_type), pointer_suffix());
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

std::unique_ptr<IRNode> IRNode::clone() {
  if (is<Block>())
    return as<Block>()->clone();
  else if (is<Stmt>())
    return as<Stmt>()->clone();
  else {
    TI_NOT_IMPLEMENTED
  }
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

int StmtFieldSNode::get_snode_id(SNode *snode) {
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
  erased = false;
  is_ptr = false;
}

Stmt::Stmt(const Stmt &stmt) : field_manager(this), fields_registered(false) {
  parent = stmt.parent;
  instance_id = instance_id_counter++;
  id = instance_id;
  erased = stmt.erased;
  is_ptr = stmt.is_ptr;
  tb = stmt.tb;
  ret_type = stmt.ret_type;
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
  int n_op = num_operands();
  for (int i = 0; i < n_op; i++) {
    if (operand(i) == old_stmt) {
      *operands[i] = new_stmt;
    }
  }
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
}

void Stmt::register_operand(Stmt *&stmt) {
  operands.push_back(&stmt);
}

void Stmt::mark_fields_registered() {
  TI_ASSERT(!fields_registered);
  fields_registered = true;
}

bool Stmt::has_operand(Stmt *stmt) const {
  for (int i = 0; i < num_operands(); i++) {
    if (*operands[i] == stmt) {
      return true;
    }
  }
  return false;
}

int Stmt::locate_operand(Stmt **stmt) {
  for (int i = 0; i < num_operands(); i++) {
    if (operands[i] == stmt) {
      return i;
    }
  }
  return -1;
}

std::string Expression::get_attribute(const std::string &key) const {
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

UnaryOpStmt::UnaryOpStmt(UnaryOpType op_type, Stmt *operand)
    : op_type(op_type), operand(operand) {
  TI_ASSERT(!operand->is<AllocaStmt>());
  cast_type = DataType::unknown;
  TI_STMT_REG_FIELDS;
}

bool UnaryOpStmt::is_cast() const {
  return unary_op_is_cast(op_type);
}

bool UnaryOpStmt::same_operation(UnaryOpStmt *o) const {
  if (op_type == o->op_type) {
    if (is_cast()) {
      return cast_type == o->cast_type;
    } else {
      return true;
    }
  }
  return false;
}

std::string UnaryOpExpression::serialize() {
  if (is_cast()) {
    std::string reint = type == UnaryOpType::cast_value ? "" : "reinterpret_";
    return fmt::format("({}{}<{}> {})", reint, unary_op_type_name(type),
                       data_type_name(cast_type), operand->serialize());
  } else {
    return fmt::format("({} {})", unary_op_type_name(type),
                       operand->serialize());
  }
}

bool UnaryOpExpression::is_cast() const {
  return unary_op_is_cast(type);
}

void UnaryOpExpression::flatten(FlattenContext *ctx) {
  operand->flatten(ctx);
  auto unary = std::make_unique<UnaryOpStmt>(type, operand->stmt);
  if (is_cast()) {
    unary->cast_type = cast_type;
  }
  stmt = unary.get();
  stmt->tb = tb;
  ctx->push_back(std::move(unary));
}

ExternalPtrStmt::ExternalPtrStmt(const LaneAttribute<Stmt *> &base_ptrs,
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

GlobalPtrStmt::GlobalPtrStmt(const LaneAttribute<SNode *> &snodes,
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

void GlobalPtrExpression::flatten(FlattenContext *ctx) {
  std::vector<Stmt *> index_stmts;
  for (int i = 0; i < (int)indices.size(); i++) {
    indices.exprs[i]->flatten(ctx);
    index_stmts.push_back(indices.exprs[i]->stmt);
  }
  if (var.is<GlobalVariableExpression>()) {
    ctx->push_back(std::make_unique<GlobalPtrStmt>(
        var.cast<GlobalVariableExpression>()->snode, index_stmts));
  } else {
    TI_ASSERT(var.is<ExternalTensorExpression>());
    var->flatten(ctx);
    ctx->push_back(std::make_unique<ExternalPtrStmt>(
        var.cast<ExternalTensorExpression>()->stmt, index_stmts));
  }
  stmt = ctx->back_stmt();
}

GetChStmt::GetChStmt(Stmt *input_ptr, int chid)
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

IRNode *FrontendContext::root() {
  return static_cast<IRNode *>(root_node.get());
}

std::unique_ptr<FrontendContext> context;

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

bool LocalLoadStmt::same_source() const {
  for (int i = 1; i < (int)ptr.size(); i++) {
    if (ptr[i].var != ptr[0].var)
      return false;
  }
  return true;
}

bool LocalLoadStmt::has_source(Stmt *alloca) const {
  for (int i = 0; i < width(); i++) {
    if (ptr[i].var == alloca)
      return true;
  }
  return false;
}

std::unique_ptr<Stmt> IfStmt::clone() const {
  auto new_stmt = std::make_unique<IfStmt>(cond);
  new_stmt->true_mask = true_mask;
  new_stmt->false_mask = false_mask;
  if (true_statements)
    new_stmt->true_statements = true_statements->clone();
  else
    new_stmt->true_statements = nullptr;
  if (false_statements)
    new_stmt->false_statements = false_statements->clone();
  else
    new_stmt->false_statements = nullptr;
  return new_stmt;
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
                         std::unique_ptr<Stmt> &&new_statement,
                         bool replace_usages) {
  VecStatement vec;
  vec.push_back(std::move(new_statement));
  replace_with(old_statement, std::move(vec), replace_usages);
}

Stmt *Block::lookup_var(const Identifier &ident) const {
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

void Block::set_statements(VecStatement &&stmts) {
  statements.clear();
  for (int i = 0; i < (int)stmts.size(); i++) {
    insert(std::move(stmts[i]), i);
  }
}

void Block::insert_before(Stmt *old_statement, VecStatement &&new_statements) {
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

void Block::replace_with(Stmt *old_statement,
                         VecStatement &&new_statements,
                         bool replace_usages) {
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
  if (new_statements.size() == 1) {
    // Keep all std::vector::iterator valid in this case.
    statements[location] = std::move(new_statements[0]);
    statements[location]->parent = this;
    return;
  }
  statements.erase(statements.begin() + location);
  for (int i = (int)new_statements.size() - 1; i >= 0; i--) {
    insert(std::move(new_statements[i]), location);
  }
}

bool Block::has_container_statements() {
  for (auto &s : statements) {
    if (s->is_container_statement())
      return true;
  }
  return false;
}

int Block::locate(Stmt *stmt) {
  for (int i = 0; i < (int)statements.size(); i++) {
    if (statements[i].get() == stmt) {
      return i;
    }
  }
  return -1;
}

std::unique_ptr<Block> Block::clone() const {
  auto new_block = std::make_unique<Block>();
  new_block->parent = parent;
  new_block->mask_var = mask_var;
  new_block->stop_gradients = stop_gradients;
  new_block->statements.reserve(size());
  for (auto &stmt : statements)
    new_block->insert(stmt->clone());
  return new_block;
}

FrontendSNodeOpStmt::FrontendSNodeOpStmt(SNodeOpType op_type,
                                         SNode *snode,
                                         const ExprGroup &indices,
                                         const Expr &val)
    : op_type(op_type), snode(snode), indices(indices.loaded()), val(val) {
  if (val.expr != nullptr) {
    TI_ASSERT(op_type == SNodeOpType::append);
    this->val.set(load_if_ptr(val));
  } else {
    TI_ASSERT(op_type != SNodeOpType::append);
  }
}

SNodeOpStmt::SNodeOpStmt(SNodeOpType op_type,
                         SNode *snode,
                         Stmt *ptr,
                         Stmt *val)
    : op_type(op_type), snode(snode), ptr(ptr), val(val) {
  width() = 1;
  element_type() = DataType::i32;
  TI_STMT_REG_FIELDS;
}

SNodeOpStmt::SNodeOpStmt(SNodeOpType op_type,
                         SNode *snode,
                         const std::vector<Stmt *> &indices)
    : op_type(op_type), snode(snode), indices(indices) {
  ptr = nullptr;
  val = nullptr;
  TI_ASSERT(op_type == SNodeOpType::is_active ||
            op_type == SNodeOpType::deactivate);
  width() = 1;
  element_type() = DataType::i32;
  TI_STMT_REG_FIELDS;
}

std::string AtomicOpExpression::serialize() {
  if (op_type == AtomicOpType::add) {
    return fmt::format("atomic_add({}, {})", dest.serialize(), val.serialize());
  } else if (op_type == AtomicOpType::sub) {
    return fmt::format("atomic_sub({}, {})", dest.serialize(), val.serialize());
  } else if (op_type == AtomicOpType::min) {
    return fmt::format("atomic_min({}, {})", dest.serialize(), val.serialize());
  } else if (op_type == AtomicOpType::max) {
    return fmt::format("atomic_max({}, {})", dest.serialize(), val.serialize());
  } else if (op_type == AtomicOpType::bit_and) {
    return fmt::format("atomic_bit_and({}, {})", dest.serialize(),
                       val.serialize());
  } else if (op_type == AtomicOpType::bit_or) {
    return fmt::format("atomic_bit_or({}, {})", dest.serialize(),
                       val.serialize());
  } else if (op_type == AtomicOpType::bit_xor) {
    return fmt::format("atomic_bit_xor({}, {})", dest.serialize(),
                       val.serialize());
  } else {
    // min/max not supported in the LLVM backend yet.
    TI_NOT_IMPLEMENTED;
  }
}

void AtomicOpExpression::flatten(FlattenContext *ctx) {
  // replace atomic sub with negative atomic add
  if (op_type == AtomicOpType::sub) {
    val.set(Expr::make<UnaryOpExpression>(UnaryOpType::neg, val));
    op_type = AtomicOpType::add;
  }
  // expand rhs
  auto expr = val;
  expr->flatten(ctx);
  if (dest.is<IdExpression>()) {  // local variable
    // emit local store stmt
    auto alloca = ctx->current_block->lookup_var(dest.cast<IdExpression>()->id);
    ctx->push_back<AtomicOpStmt>(op_type, alloca, expr->stmt);
  } else {  // global variable
    TI_ASSERT(dest.is<GlobalPtrExpression>());
    auto global_ptr = dest.cast<GlobalPtrExpression>();
    global_ptr->flatten(ctx);
    ctx->push_back<AtomicOpStmt>(op_type, ctx->back_stmt(), expr->stmt);
  }
  stmt = ctx->back_stmt();
}

std::string SNodeOpExpression::serialize() {
  if (value.expr) {
    return fmt::format("{}({}, [{}], {})", snode_op_type_name(op_type),
                       snode->get_node_type_name_hinted(), indices.serialize(),
                       value.serialize());
  } else {
    return fmt::format("{}({}, [{}])", snode_op_type_name(op_type),
                       snode->get_node_type_name_hinted(), indices.serialize());
  }
}

void SNodeOpExpression::flatten(FlattenContext *ctx) {
  std::vector<Stmt *> indices_stmt;
  for (int i = 0; i < (int)indices.size(); i++) {
    indices[i]->flatten(ctx);
    indices_stmt.push_back(indices[i]->stmt);
  }
  if (op_type == SNodeOpType::is_active) {
    // is_active cannot be lowered all the way to a global pointer.
    // It should be lowered into a pointer to parent and an index.
    TI_ERROR_IF(snode->type != SNodeType::pointer &&
                    snode->type != SNodeType::hash &&
                    snode->type != SNodeType::bitmasked,
                "ti.is_active only works on pointer, hash or bitmasked nodes.");
    ctx->push_back<SNodeOpStmt>(SNodeOpType::is_active, snode, indices_stmt);
  } else {
    auto ptr = ctx->push_back<GlobalPtrStmt>(snode, indices_stmt);
    if (op_type == SNodeOpType::append) {
      value->flatten(ctx);
      ctx->push_back<SNodeOpStmt>(SNodeOpType::append, snode, ptr,
                                  ctx->back_stmt());
      TI_ERROR_IF(snode->type != SNodeType::dynamic,
                  "ti.append only works on dynamic nodes.");
      TI_ERROR_IF(snode->ch.size() != 1,
                  "ti.append only works on single-child dynamic nodes.");
      TI_ERROR_IF(data_type_size(snode->ch[0]->dt) != 4,
                  "ti.append only works on i32/f32 nodes.");
    } else if (op_type == SNodeOpType::length) {
      ctx->push_back<SNodeOpStmt>(SNodeOpType::length, snode, ptr, nullptr);
    }
  }
  stmt = ctx->back_stmt();
}

std::unique_ptr<ConstStmt> ConstStmt::copy() {
  return std::make_unique<ConstStmt>(val);
}

RangeForStmt::RangeForStmt(Stmt *loop_var,
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

std::unique_ptr<Stmt> RangeForStmt::clone() const {
  auto new_stmt = std::make_unique<RangeForStmt>(
      loop_var, begin, end, body->clone(), vectorize, parallelize,
      block_dim, strictly_serialized);
  new_stmt->reversed = reversed;
  return new_stmt;
}

StructForStmt::StructForStmt(std::vector<Stmt *> loop_vars,
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

std::unique_ptr<Stmt> StructForStmt::clone() const {
  auto new_stmt = std::make_unique<StructForStmt>(
      loop_vars, snode, body->clone(), vectorize, parallelize, block_dim);
  new_stmt->scratch_opt = scratch_opt;
  return new_stmt;
}

std::unique_ptr<Stmt> FuncBodyStmt::clone() const {
  return std::make_unique<FuncBodyStmt>(funcid, body->clone());
}

std::unique_ptr<Stmt> WhileStmt::clone() const {
  auto new_stmt = std::make_unique<WhileStmt>(body->clone());
  new_stmt->mask = mask;
  return new_stmt;
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

void Stmt::infer_type() {
  irpass::typecheck(this);
}

TLANG_NAMESPACE_END
