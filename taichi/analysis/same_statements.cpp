#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/async_utils.h"
#include "taichi/program/ir_bank.h"
#include <unordered_map>
#include <unordered_set>
#include <utility>

TLANG_NAMESPACE_BEGIN

// Compare if two IRNodes are equivalent.
class IRNodeComparator : public IRVisitor {
 private:
  IRNode *other_node;
  // map the id from this node to the other node
  std::unordered_map<int, int> id_map;

  bool recursively_check_;
  bool check_same_value_;

  std::unordered_set<AsyncState> possibly_modified_states_;
  bool all_states_can_be_modified_;
  IRBank *ir_bank_;

 public:
  bool same;

  explicit IRNodeComparator(
      IRNode *other_node,
      const std::optional<std::unordered_map<int, int>> &id_map,
      bool check_same_value,
      const std::optional<std::unordered_set<AsyncState>>
          &possibly_modified_states,
      IRBank *ir_bank)
      : other_node(other_node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    same = true;
    if (id_map.has_value()) {
      recursively_check_ = true;
      this->id_map = id_map.value();
    } else {
      recursively_check_ = false;
    }
    if (possibly_modified_states.has_value()) {
      TI_ASSERT_INFO(check_same_value,
                     "The parameter possibly_modified_states "
                     "is only supported when check_same_value is true");
      TI_ASSERT_INFO(ir_bank,
                     "The parameter possibly_modified_states "
                     "requires ir_bank")
      all_states_can_be_modified_ = false;
      this->possibly_modified_states_ = possibly_modified_states.value();
    } else {
      all_states_can_be_modified_ = true;
    }
    check_same_value_ = check_same_value;
    ir_bank_ = ir_bank;
  }

  void map_id(int this_id, int other_id) {
    auto it = id_map.find(this_id);
    if (it == id_map.end()) {
      id_map[this_id] = other_id;
    } else if (it->second != other_id) {
      same = false;
    }
  }

  void check_mapping(Stmt *this_stmt, Stmt *other_stmt) {
    // get the corresponding id in the other node
    // and check if it is other_stmt->id
    auto it = id_map.find(this_stmt->id);
    if (it != id_map.end()) {
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
      id_map[this_stmt->id] = other_stmt->id;
    } else {
      // recursively check them
      IRNode *backup_other_node = other_node;
      other_node = other_stmt;
      this_stmt->accept(this);
      other_node = backup_other_node;
    }
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
    if (verbose)
      std::cout << "checking " << stmt->id << std::endl;
    // type check
    if (typeid(*other_node) != typeid(*stmt)) {
      same = false;
      return;
    }
    if (verbose)
      std::cout << "qqq" << std::endl;
    auto other = other_node->as<Stmt>();

    // If two identical statements can have different values, return false.
    // TODO: two identical GlobalPtrStmts cannot have different values,
    //  but GlobalPtrStmt::common_statement_eliminable() is false.
    if (check_same_value_ && stmt != other && !stmt->is_container_statement() &&
        !stmt->common_statement_eliminable() && !stmt->is<GlobalPtrStmt>()) {
      if (all_states_can_be_modified_) {
        same = false;
        return;
      } else {
        // "break" all branches that do not result in "same = false"
        do {
          if (auto global_load = stmt->cast<GlobalLoadStmt>()) {
            if (auto global_ptr = global_load->ptr->cast<GlobalPtrStmt>()) {
              TI_ASSERT(global_ptr->width() == 1);
              if (possibly_modified_states_.count(ir_bank_->get_async_state(
                      global_ptr->snodes[0], AsyncState::Type::value)) == 0) {
                break;
              }
            }
            // TODO: other cases?
          } else if (auto global_store = stmt->cast<GlobalStoreStmt>()) {
            if (auto global_ptr = global_store->ptr->cast<GlobalPtrStmt>()) {
              TI_ASSERT(global_ptr->width() == 1);
              if (possibly_modified_states_.count(ir_bank_->get_async_state(
                      global_ptr->snodes[0], AsyncState::Type::value)) == 0) {
                break;
              }
            }
          }
          same = false;
          return;
        } while (false);
      }
    }
    if (verbose)
      std::cout << "www" << std::endl;
    // Note that we do not need to test !stmt2->common_statement_eliminable()
    // because if this condition does not hold,
    // same_statements(stmt1, stmt2) returns false anyway.

    // field check
    if (check_same_value_ && stmt->is<GlobalPtrStmt>()) {
      if (verbose)
        std::cout << "eee" << std::endl;
      // Special case: we do not care the "activate" field when checking
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
    } else {
      if (verbose)
        std::cout << "rrr" << std::endl;
      if (!stmt->field_manager.equal(other->field_manager)) {
        same = false;
        return;
      }
    }
    if (verbose)
      std::cout << "ttt" << std::endl;

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
    if (verbose)
      std::cout << "yyy" << std::endl;

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
  bool verbose{false};

  static bool run(IRNode *root1,
                  IRNode *root2,
                  const std::optional<std::unordered_map<int, int>> &id_map,
                  bool check_same_value,
                  const std::optional<std::unordered_set<AsyncState>>
                      &possibly_modified_states,
                  IRBank *ir_bank) {
    IRNodeComparator comparator(root2, id_map, check_same_value,
                                possibly_modified_states, ir_bank);
//    if (check_same_value && id_map.has_value())
//      comparator.verbose = true;
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
                               /*check_same_value=*/false, std::nullopt,
                               /*ir_bank=*/nullptr);
}
bool same_value(Stmt *stmt1,
                Stmt *stmt2,
                const AsyncStateSet &possibly_modified_states,
                IRBank *ir_bank,
                const std::optional<std::unordered_map<int, int>> &id_map) {
  // Test if two statements definitely have the same value.
  if (stmt1 == stmt2)
    return true;
  if (!stmt1 || !stmt2)
    return false;
  return IRNodeComparator::run(
      stmt1, stmt2, id_map, /*check_same_value=*/true,
      std::make_optional<std::unordered_set<AsyncState>>(
          possibly_modified_states.s),
      ir_bank);
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
                               /*check_same_value=*/true, std::nullopt,
                               /*ir_bank=*/nullptr);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
