
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
  TI_ASSERT(!stack.empty());
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
    list->parent_stmt = get_last_stmt();
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

IRNode *IRNode::get_ir_root() {
  auto node = this;
  while (node->get_parent()) {
    node = node->get_parent();
  }
  return node;
}

Kernel *IRNode::get_kernel() const {
  return const_cast<IRNode *>(this)->get_ir_root()->kernel;
}

CompileConfig &IRNode::get_config() const {
  return get_kernel()->program.config;
}

std::unique_ptr<IRNode> IRNode::clone() {
  std::unique_ptr<IRNode> new_irnode;
  if (is<Block>())
    new_irnode = as<Block>()->clone();
  else if (is<Stmt>())
    new_irnode = as<Stmt>()->clone();
  else {
    TI_NOT_IMPLEMENTED
  }
  new_irnode->kernel = kernel;
  return new_irnode;
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
}

Stmt::Stmt(const Stmt &stmt) : field_manager(this), fields_registered(false) {
  parent = stmt.parent;
  instance_id = instance_id_counter++;
  id = instance_id;
  erased = stmt.erased;
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
  irpass::replace_all_usages_with(nullptr, this, new_stmt);
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
  if (ret_type->is_primitive(PrimitiveTypeID::unknown))
    return "";
  else
    return fmt::format("<{}> ", ret_type.to_string());
}

std::string Stmt::type() {
  StatementTypeNameVisitor v;
  this->accept(&v);
  return v.type_name;
}

IRNode *Stmt::get_parent() const {
  return parent;
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

Stmt *Block::insert(std::unique_ptr<Stmt> &&stmt, int location) {
  auto stmt_ptr = stmt.get();
  stmt->parent = this;
  if (location == -1) {
    statements.push_back(std::move(stmt));
  } else {
    statements.insert(statements.begin() + location, std::move(stmt));
  }
  return stmt_ptr;
}

Stmt *Block::insert(VecStatement &&stmt, int location) {
  Stmt *stmt_ptr = nullptr;
  if (stmt.size()) {
    stmt_ptr = stmt.back().get();
  }
  if (location == -1) {
    location = (int)statements.size();
  }
  for (int i = 0; i < stmt.size(); i++) {
    insert(std::move(stmt[i]), location + i);
  }
  return stmt_ptr;
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
  auto ptr = local_var_to_stmt.find(ident);
  if (ptr != local_var_to_stmt.end()) {
    return ptr->second;
  } else {
    if (parent_block()) {
      return parent_block()->lookup_var(ident);
    } else {
      return nullptr;
    }
  }
}

Stmt *Block::mask() {
  if (mask_var)
    return mask_var;
  else if (parent_block() == nullptr) {
    return nullptr;
  } else {
    return parent_block()->mask();
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

void Block::insert_after(Stmt *old_statement, VecStatement &&new_statements) {
  int location = -1;
  for (int i = 0; i < (int)statements.size(); i++) {
    if (old_statement == statements[i].get()) {
      location = i + 1;
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
  if (replace_usages && !new_statements.stmts.empty())
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

Block *Block::parent_block() const {
  if (parent_stmt == nullptr)
    return nullptr;
  return parent_stmt->parent;
}

IRNode *Block::get_parent() const {
  return parent_stmt;
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
  new_block->parent_stmt = parent_stmt;
  new_block->mask_var = mask_var;
  new_block->stop_gradients = stop_gradients;
  new_block->statements.reserve(size());
  for (auto &stmt : statements)
    new_block->insert(stmt->clone());
  return new_block;
}

DelayedIRModifier::~DelayedIRModifier() {
  TI_ASSERT(to_insert_before.empty());
  TI_ASSERT(to_insert_after.empty());
  TI_ASSERT(to_erase.empty());
  TI_ASSERT(to_replace_with.empty());
}

void DelayedIRModifier::erase(Stmt *stmt) {
  to_erase.push_back(stmt);
}

void DelayedIRModifier::insert_before(Stmt *old_statement,
                                      std::unique_ptr<Stmt> new_statements) {
  to_insert_before.emplace_back(old_statement,
                                VecStatement(std::move(new_statements)));
}

void DelayedIRModifier::insert_before(Stmt *old_statement,
                                      VecStatement &&new_statements) {
  to_insert_before.emplace_back(old_statement, std::move(new_statements));
}

void DelayedIRModifier::insert_after(Stmt *old_statement,
                                     std::unique_ptr<Stmt> new_statements) {
  to_insert_after.emplace_back(old_statement,
                               VecStatement(std::move(new_statements)));
}

void DelayedIRModifier::insert_after(Stmt *old_statement,
                                     VecStatement &&new_statements) {
  to_insert_after.emplace_back(old_statement, std::move(new_statements));
}

void DelayedIRModifier::replace_with(Stmt *stmt,
                                     VecStatement &&new_statements,
                                     bool replace_usages) {
  to_replace_with.emplace_back(stmt, std::move(new_statements), replace_usages);
}

bool DelayedIRModifier::modify_ir() {
  if (to_insert_before.empty() && to_insert_after.empty() && to_erase.empty() &&
      to_replace_with.empty())
    return false;
  for (auto &i : to_insert_before) {
    i.first->parent->insert_before(i.first, std::move(i.second));
  }
  to_insert_before.clear();
  for (auto &i : to_insert_after) {
    i.first->parent->insert_after(i.first, std::move(i.second));
  }
  to_insert_after.clear();
  for (auto &stmt : to_erase) {
    stmt->parent->erase(stmt);
  }
  to_erase.clear();
  for (auto &i : to_replace_with) {
    std::get<0>(i)->replace_with(std::move(std::get<1>(i)), std::get<2>(i));
  }
  to_replace_with.clear();
  return true;
}

LocalAddress::LocalAddress(Stmt *var, int offset) : var(var), offset(offset) {
  TI_ASSERT(var->is<AllocaStmt>());
}

void Stmt::infer_type() {
  irpass::type_check(this);
}

TLANG_NAMESPACE_END
