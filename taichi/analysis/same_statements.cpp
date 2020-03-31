#include "taichi/ir/ir.h"
#include <map>
#include <set>

TLANG_NAMESPACE_BEGIN

#define DEFINE_TYPE_CHECK(Type)       \
  if (!other_node->is<Type>()) {      \
    same = false;                     \
    return;                           \
  }                                   \
  auto other = other_node->as<Type>()  // semicolon intentionally omitted

#define DEFINE_OPERAND_CHECK                                           \
  if (stmt->num_operands() != other->num_operands()) {                 \
    same = false;                                                      \
    return;                                                            \
  }                                                                    \
  for (int i = 0; i < stmt->num_operands(); i++) {                     \
    if (get_other_id(stmt->operand(i)->id) != other->operand(i)->id) { \
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
    DEFINE_TYPE_CHECK(Block);
    if (stmt_list->statements.size() != other->statements.size()) {
      same = false;
      return;
    }
    for (int i = 0; i < (int)stmt_list->statements.size(); i++) {
      other_node = other->statements[i].get();
      stmt_list->statements[i]->accept(this);
      if (!same)
        break;
    }
    other_node = other;
  }

  /*void visit(FrontendBreakStmt *stmt) override {
    DEFINE_TYPE_CHECK(FrontendBreakStmt);
  }

  void visit(FrontendAssignStmt *assign) override {
    DEFINE_TYPE_CHECK(FrontendAssignStmt);
    if (assign->lhs.serialize() != other->lhs.serialize()
        || assign->rhs.serialize() != other->rhs.serialize())
      same = false;
  }

  void visit(FrontendAllocaStmt *alloca) override {
    DEFINE_TYPE_CHECK(FrontendAllocaStmt);
    if (alloca->type_hint() != other->type_hint()) {
      same = false;
      return;
    }
    if (alloca->ident.name() != other->ident.name()) {
      same = false;
      return;
    }
    map_id(alloca->id, other->id);
  }*/

  void visit(AssertStmt *stmt) override {
    DEFINE_TYPE_CHECK(AssertStmt);
    DEFINE_OPERAND_CHECK
    DEFINE_FIELD_CHECK(text)
    DEFINE_MAP_ID
  }

  void visit(SNodeOpStmt *stmt) override {
    DEFINE_TYPE_CHECK(SNodeOpStmt);
    DEFINE_OPERAND_CHECK
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(op_type)
    DEFINE_FIELD_CHECK(snode->id)
    DEFINE_MAP_ID
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
