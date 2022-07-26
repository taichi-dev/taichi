#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
#include <unordered_map>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// Compare if two IRNodes are equivalent.
class IRNodeComparator : public IRVisitor {
 private:
  IRNode *other_node_;
  // map the id from this node to the other node
  std::unordered_map<int, int> id_map_;

  bool recursively_check_;

  // Compare if two IRNodes definitely have the same value instead.
  // When this is true, it's weaker in the sense that we don't require the
  // activate field in the GlobalPtrStmt to be the same, but stronger in the
  // sense that we require the value to be the same (especially stronger in
  // GlobalLoadStmt, RandStmt, etc.).
  bool check_same_value_;

 public:
  bool same;

  explicit IRNodeComparator(
      IRNode *other_node,
      const std::optional<std::unordered_map<int, int>> &id_map,
      bool check_same_value)
      : other_node_(other_node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    same = true;
    if (id_map.has_value()) {
      recursively_check_ = true;
      this->id_map_ = id_map.value();
    } else {
      recursively_check_ = false;
    }
    check_same_value_ = check_same_value;
  }

  void map_id(int this_id, int other_id) {
    auto it = id_map_.find(this_id);
    if (it == id_map_.end()) {
      id_map_[this_id] = other_id;
    } else if (it->second != other_id) {
      same = false;
    }
  }

  void check_mapping(Stmt *this_stmt, Stmt *other_stmt) {
    // get the corresponding id in the other node
    // and check if it is other_stmt->id
    auto it = id_map_.find(this_stmt->id);
    if (it != id_map_.end()) {
      if (it->second != other_stmt->id) {
        same = false;
      }
      return;
    }
    if (!recursively_check_) {
      // use identity mapping if not found
      if (this_stmt->id != other_stmt->id) {
        same = false;
      }
      id_map_[this_stmt->id] = other_stmt->id;
    } else {
      // recursively check them
      IRNode *backup_other_node = other_node_;
      other_node_ = other_stmt;
      this_stmt->accept(this);
      other_node_ = backup_other_node;
    }
  }

  void visit(Block *stmt_list) override {
    if (!other_node_->is<Block>()) {
      same = false;
      return;
    }

    auto other = other_node_->as<Block>();
    if (stmt_list->size() != other->size()) {
      same = false;
      return;
    }
    for (int i = 0; i < (int)stmt_list->size(); i++) {
      other_node_ = other->statements[i].get();
      stmt_list->statements[i]->accept(this);
      if (!same)
        break;
    }
    other_node_ = other;
  }

  void basic_check(Stmt *stmt) {
    // type check
    if (typeid(*other_node_) != typeid(*stmt)) {
      same = false;
      return;
    }
    auto other = other_node_->as<Stmt>();
    if (stmt == other) {
      return;
    }

    // If two identical statements can have different values, return false.
    // TODO: actually the condition should be "can stmt be an operand of
    //  another statement?"
    const bool stmt_has_value = !stmt->is_container_statement();
    // TODO: We want to know if two identical statements of the type same as
    //  stmt can have different values. In most cases, this property is the
    //  same as Stmt::common_statement_eliminable(). However, two identical
    //  GlobalPtrStmts cannot have different values, although
    //  GlobalPtrStmt::common_statement_eliminable() is false.
    const bool identical_stmts_can_have_different_value =
        stmt_has_value && !stmt->common_statement_eliminable() &&
        !stmt->is<GlobalPtrStmt>();
    // Note that we do not need to test !stmt2->common_statement_eliminable()
    // because if this condition does not hold,
    // same_value(stmt1, stmt2) returns false anyway.
    if (check_same_value_ && identical_stmts_can_have_different_value) {
      same = false;
      return;
    }

    bool field_checked = false;
    if (check_same_value_) {
      if (stmt->is<GlobalPtrStmt>()) {
        // Special case: we do not care about the "activate" field when checking
        // whether two global pointers share the same value.
        // And we cannot use irpass::analysis::definitely_same_address()
        // directly because that function does not support id_map.

        // TODO: Update this part if GlobalPtrStmt comes to have more fields
        TI_ASSERT(stmt->width() == 1);
        if (stmt->as<GlobalPtrStmt>()->snodes[0]->id !=
            other->as<GlobalPtrStmt>()->snodes[0]->id) {
          same = false;
          return;
        }
        field_checked = true;
      } else if (stmt->is<LoopUniqueStmt>()) {
        // Special case: we do not care the "covers" field when checking
        // whether two LoopUniqueStmts share the same value.
        field_checked = true;
      } else if (stmt->is<RangeAssumptionStmt>()) {
        // Special case: we do not care the "low, high" fields when checking
        // whether two RangeAssumptionStmts share the same value.
        field_checked = true;
      }
    }
    if (!field_checked) {
      // field check
      if (!stmt->field_manager.equal(other->field_manager)) {
        same = false;
        return;
      }
    }

    bool operand_checked = false;
    if (check_same_value_) {
      if (stmt->is<RangeAssumptionStmt>()) {
        // Special case: we do not care about the "base" operand when checking
        // whether two RangeAssumptionStmts share the same value.
        check_mapping(stmt->as<RangeAssumptionStmt>()->input,
                      other->as<RangeAssumptionStmt>()->input);
        operand_checked = true;
      }
    }
    if (!operand_checked) {
      // operand check
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
        check_mapping(stmt->operand(i), other->operand(i));
      }
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
    auto other = other_node_->as<IfStmt>();
    if (stmt->true_statements) {
      if (!other->true_statements) {
        same = false;
        return;
      }
      other_node_ = other->true_statements.get();
      stmt->true_statements->accept(this);
      other_node_ = other;
    }
    if (stmt->false_statements && same) {
      if (!other->false_statements) {
        same = false;
        return;
      }
      other_node_ = other->false_statements.get();
      stmt->false_statements->accept(this);
      other_node_ = other;
    }
  }

  void visit(WhileStmt *stmt) override {
    basic_check(stmt);
    if (!same)
      return;
    auto other = other_node_->as<WhileStmt>();
    other_node_ = other->body.get();
    stmt->body->accept(this);
    other_node_ = other;
  }

  void visit(RangeForStmt *stmt) override {
    basic_check(stmt);
    if (!same)
      return;
    auto other = other_node_->as<RangeForStmt>();
    other_node_ = other->body.get();
    stmt->body->accept(this);
    other_node_ = other;
  }

  void visit(StructForStmt *stmt) override {
    basic_check(stmt);
    if (!same)
      return;
    auto other = other_node_->as<StructForStmt>();
    other_node_ = other->body.get();
    stmt->body->accept(this);
    other_node_ = other;
  }

  void visit(OffloadedStmt *stmt) override {
    basic_check(stmt);
    if (!same)
      return;
    auto other = other_node_->as<OffloadedStmt>();
    if (stmt->has_body()) {
      TI_ASSERT(stmt->body);
      TI_ASSERT(other->body);
      other_node_ = other->body.get();
      stmt->body->accept(this);
      other_node_ = other;
    }
  }

  static bool run(IRNode *root1,
                  IRNode *root2,
                  const std::optional<std::unordered_map<int, int>> &id_map,
                  bool check_same_value) {
    IRNodeComparator comparator(root2, id_map, check_same_value);
    root1->accept(&comparator);
    return comparator.same;
  }
};

namespace irpass::analysis {
bool same_statements(
    IRNode *root1,
    IRNode *root2,
    const std::optional<std::unordered_map<int, int>> &id_map) {
  // When id_map is std::nullopt by default, this function tests if
  // root1 and root2 are the same, i.e., have the same type,
  // the same operands and the same fields.
  // If root1 and root2 are container statements or statement blocks,
  // this function traverses the contents correspondingly.
  // Two operands are considered the same if they have the same id
  // and do not belong to either root, or they belong to root1 and root2
  // at the same position in the roots.
  //
  // For example, same_statements(block1, block2, std::nullopt) is true:
  // <i32> $1 = ...
  // block1 : {
  //   <i32> $2 = const [1]
  //   <i32> $3 = add $1 $2
  // }
  // block2 : {
  //   <i32> $4 = const [1]
  //   <i32> $5 = add $1 $4
  // }
  //
  // If id_map is not std::nullopt, this function also recursively
  // check the operands until ids in the id_map are reached.
  // id_map is an id map from root1 to root2.
  //
  // In the above example, same_statements($3, $5, std::nullopt) is false
  // but same_statements($3, $5, (an empty map)) is true.
  //
  // In the following example, same_statements($3, $6, id_map) is true
  // iff id_map[1] == 4 && id_map[2] == 5:
  // <i32> $3 = add $1 $2
  // <i32> $6 = add $4 $5
  //
  // Null pointers as IRNodes are defined to be NOT the same as any other
  // IRNode, except for another nullptr IRNode.
  if (root1 == root2)
    return true;
  if (!root1 || !root2)
    return false;
  return IRNodeComparator::run(root1, root2, id_map,
                               /*check_same_value=*/false);
}

bool same_value(Stmt *stmt1,
                Stmt *stmt2,
                const std::optional<std::unordered_map<int, int>> &id_map) {
  // Test if two statements definitely have the same value.
  if (stmt1 == stmt2)
    return true;
  if (!stmt1 || !stmt2)
    return false;
  return IRNodeComparator::run(stmt1, stmt2, id_map,
                               /*check_same_value=*/true);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
