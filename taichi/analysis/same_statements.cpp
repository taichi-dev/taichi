#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include <unordered_map>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// Compare if two IRNodes are equivalent.
class IRNodeComparator : public IRVisitor {
 private:
  IRNode *other_node;
  // map the id from this node to the other node
  std::unordered_map<int, int> id_map;
  // ids which don't belong to either node
  std::unordered_set<int> captured_id;

 public:
  bool same;

  explicit IRNodeComparator(IRNode *other_node) : other_node(other_node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    same = true;
  }

  void map_id(int this_id, int other_id) {
    if (captured_id.find(this_id) != captured_id.end() ||
        captured_id.find(other_id) != captured_id.end()) {
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
    if (!other_node->is<Block>()) {
      same = false;
      return;
    }

    auto other = other_node->as<Block>();
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

  void basic_check(Stmt *stmt) {
    // type check
    if (typeid(*other_node) != typeid(*stmt)) {
      same = false;
      return;
    }

    // operand check
    auto other = other_node->as<Stmt>();
    if (stmt->num_operands() != other->num_operands()) {
      same = false;
      return;
    }
    for (int i = 0; i < stmt->num_operands(); i++) {
      if ((stmt->operand(i) == nullptr) != (other->operand(i) == nullptr)) {
        same = false;
        return;
      }
      if (stmt->operand(i) == nullptr)
        continue;
      if (get_other_id(stmt->operand(i)->id) != other->operand(i)->id) {
        same = false;
        return;
      }
    }

    // field check
    if (!stmt->field_manager.equal(other->field_manager)) {
      same = false;
      return;
    }

    map_id(stmt->id, other->id);
  }

  void visit(Stmt *stmt) override {
    basic_check(stmt);
  }

  void visit(IfStmt *stmt) override {
    basic_check(stmt);
    if (!same)
      return;
    auto other = other_node->as<IfStmt>();
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

  void visit(FuncBodyStmt *stmt) override {
    basic_check(stmt);
    if (!same)
      return;
    auto other = other_node->as<FuncBodyStmt>();
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(WhileStmt *stmt) override {
    basic_check(stmt);
    if (!same)
      return;
    auto other = other_node->as<WhileStmt>();
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(RangeForStmt *stmt) override {
    basic_check(stmt);
    if (!same)
      return;
    auto other = other_node->as<RangeForStmt>();
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(StructForStmt *stmt) override {
    basic_check(stmt);
    if (!same)
      return;
    auto other = other_node->as<StructForStmt>();
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(OffloadedStmt *stmt) override {
    basic_check(stmt);
    if (!same)
      return;
    auto other = other_node->as<OffloadedStmt>();
    if (stmt->has_body()) {
      TI_ASSERT(stmt->body);
      TI_ASSERT(other->body);
      other_node = other->body.get();
      stmt->body->accept(this);
      other_node = other;
    }
  }

  static bool run(IRNode *root1, IRNode *root2) {
    IRNodeComparator comparator(root2);
    root1->accept(&comparator);
    return comparator.same;
  }
};

namespace irpass::analysis {
bool same_statements(IRNode *root1, IRNode *root2) {
  return IRNodeComparator::run(root1, root2);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
