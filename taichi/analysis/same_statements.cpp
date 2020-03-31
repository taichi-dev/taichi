#include "taichi/ir/ir.h"
#include <map>
#include <set>

TLANG_NAMESPACE_BEGIN

#define DEFINE_TYPE_CHECK(Type)       \
  if (!other_node->is<Type>()) {      \
    same = false;                     \
    return;                           \
  }                                   \
  auto other = other_node->as<Type>();

#define DEFINE_OPERAND_CHECK                                           \
  if (stmt->num_operands() != other->num_operands()) {                 \
    same = false;                                                      \
    return;                                                            \
  }                                                                    \
  for (int i = 0; i < stmt->num_operands(); i++) {                     \
    if (get_other_id(stmt->Stmt::operand(i)->id)                       \
        != other->Stmt::operand(i)->id) {                              \
      /*to distinguish from UnaryOpStmt::operand*/                     \
      same = false;                                                    \
      return;                                                          \
    }                                                                  \
  }

#define DEFINE_FIELD_CHECK(field)    \
  if (stmt->field != other->field) { \
    same = false;                    \
    return;                          \
  }

#define DEFINE_MAP_ID map_id(stmt->id, other->id);

#define DEFINE_BASIC_CHECK(Type) \
  DEFINE_TYPE_CHECK(Type)        \
  DEFINE_OPERAND_CHECK           \
  DEFINE_MAP_ID

// Compare if two IRNodes are equivalent.
class IRNodeComparator : public IRVisitor {
 private:
  IRNode *other_node;
  std::map<int, int> id_map; // map the id from this node to the other node
  std::set<int> captured_id; // ids which don't belong to either node

 public:
  bool same;

  explicit IRNodeComparator(IRNode *other_node) : other_node(other_node) {
    same = true;
  }

  void map_id(int this_id, int other_id) {
    if (captured_id.find(this_id) != captured_id.end()
        || captured_id.find(other_id) != captured_id.end()) {
      same = false;
      return;
    }
    auto it = id_map.find(this_id);
    if (it == id_map.end()) {
      id_map[this_id] = other_id;
    } else if (it->second != other_id) {
      same = false;
    }
  }

  int get_other_id(int this_id) {
    // get the corresponding id in the other node
    auto it = id_map.find(this_id);
    if (it != id_map.end()) {
      return it->second;
    }
    // if not found, should be captured
    // (What if this_id belongs to the other node? Ignoring this case here.)
    if (captured_id.find(this_id) == captured_id.end()) {
      captured_id.insert(this_id);
    }
    return this_id;
  }

  void visit(Block *stmt_list) override {
    DEFINE_TYPE_CHECK(Block)
    if (stmt_list->size() != other->size()) {
      same = false;
      return;
    }
    for (int i = 0; i < (int)stmt_list->size(); i++) {
      other_node = other->statements[i].get();
      stmt_list->statements[i]->accept(this);
      if (!same)
        break;
    }
    other_node = other;
  }

  void visit(AssertStmt *stmt) override {
    DEFINE_BASIC_CHECK(AssertStmt)
    DEFINE_FIELD_CHECK(text)
  }

  void visit(SNodeOpStmt *stmt) override {
    DEFINE_BASIC_CHECK(SNodeOpStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(op_type)
    DEFINE_FIELD_CHECK(snode->id)
  }

  void visit(AllocaStmt *stmt) override {
    DEFINE_BASIC_CHECK(AllocaStmt)
    DEFINE_FIELD_CHECK(type_hint())
  }

  void visit(RandStmt *stmt) override {
    DEFINE_BASIC_CHECK(RandStmt)
    DEFINE_FIELD_CHECK(type_hint())
  }

  void visit(UnaryOpStmt *stmt) override {
    DEFINE_BASIC_CHECK(UnaryOpStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(op_type)
  }

  void visit(BinaryOpStmt *stmt) override {
    DEFINE_BASIC_CHECK(BinaryOpStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(op_type)
  }

  void visit(TernaryOpStmt *stmt) override {
    DEFINE_BASIC_CHECK(TernaryOpStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(op_type)
  }

  void visit(AtomicOpStmt *stmt) override {
    DEFINE_BASIC_CHECK(AtomicOpStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(op_type)
  }

  void visit(IfStmt *stmt) override {
    DEFINE_BASIC_CHECK(IfStmt)
    if (stmt->true_statements) {
      if (!other->true_statements) {
        same = false;
        return;
      }
      other_node = other->true_statements.get();
      stmt->true_statements->accept(this);
      other_node = other;
    }
    if (stmt->false_statements && same) {
      if (!other->false_statements) {
        same = false;
        return;
      }
      other_node = other->false_statements.get();
      stmt->false_statements->accept(this);
      other_node = other;
    }
  }

  void visit(PrintStmt *stmt) override {
    DEFINE_BASIC_CHECK(PrintStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(str)
  }

  void visit(ConstStmt *stmt) override {
    DEFINE_BASIC_CHECK(ConstStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(val.serialize(
        [](const TypedConstant &t) { return t.stringify(); }, "["))
  }

  void visit(WhileControlStmt *stmt) override {
    DEFINE_BASIC_CHECK(WhileControlStmt)
  }

  void visit(FuncCallStmt *stmt) override {
    DEFINE_BASIC_CHECK(FuncCallStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(funcid)
  }

  void visit(FuncBodyStmt *stmt) override {
    DEFINE_BASIC_CHECK(FuncBodyStmt)
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(WhileStmt *stmt) override {
    DEFINE_BASIC_CHECK(WhileStmt)
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(RangeForStmt *stmt) override {
    DEFINE_BASIC_CHECK(RangeForStmt)
    DEFINE_FIELD_CHECK(reversed)
    DEFINE_FIELD_CHECK(vectorize)
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(StructForStmt *stmt) override {
    DEFINE_BASIC_CHECK(StructForStmt)
    DEFINE_FIELD_CHECK(snode->id)
    DEFINE_FIELD_CHECK(vectorize)
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  static bool run(IRNode *root1, IRNode *root2) {
    IRNodeComparator comparator(root2);
    root1->accept(&comparator);
    return comparator.same;
  }
};

namespace irpass {
bool same_statements(IRNode *root1, IRNode *root2) {
  return IRNodeComparator::run(root1, root2);
}
}  // namespace irpass

TLANG_NAMESPACE_END
