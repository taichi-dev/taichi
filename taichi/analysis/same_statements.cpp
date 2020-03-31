#include "taichi/ir/ir.h"
#include <unordered_map>
#include <unordered_set>

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

#define DEFINE_SNODE_CHECK(snode)                                \
  if (get_snode_id(stmt->snode) != get_snode_id(other->snode)) { \
    same = false;                                                \
    return;                                                      \
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
  // map the id from this node to the other node
  std::unordered_map<int, int> id_map;
  // ids which don't belong to either node
  std::unordered_set<int> captured_id;

 public:
  bool same;

  explicit IRNodeComparator(IRNode *other_node) : other_node(other_node) {
    same = true;
  }

  static int get_snode_id(SNode *snode) {
    if (snode == nullptr)
      return -1;
    return snode->id;
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
    DEFINE_SNODE_CHECK(snode)
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
    DEFINE_SNODE_CHECK(snode)
    DEFINE_FIELD_CHECK(vectorize)
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(GlobalPtrStmt *stmt) override {
    DEFINE_BASIC_CHECK(GlobalPtrStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(width())
    DEFINE_FIELD_CHECK(activate)
    for (int l = 0; l < stmt->width(); l++) {
      DEFINE_SNODE_CHECK(snodes[l])
    }
  }

  void visit(ArgLoadStmt *stmt) override {
    DEFINE_BASIC_CHECK(ArgLoadStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(arg_id)
  }

  void visit(ArgStoreStmt *stmt) override {
    DEFINE_BASIC_CHECK(ArgStoreStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(arg_id)
  }

  void visit(LocalLoadStmt *stmt) override {
    DEFINE_BASIC_CHECK(LocalLoadStmt)
    DEFINE_FIELD_CHECK(type_hint())
  }

  void visit(LocalStoreStmt *stmt) override {
    DEFINE_BASIC_CHECK(LocalStoreStmt)
    DEFINE_FIELD_CHECK(type_hint())
  }

  void visit(GlobalLoadStmt *stmt) override {
    DEFINE_BASIC_CHECK(GlobalLoadStmt)
    DEFINE_FIELD_CHECK(type_hint())
  }

  void visit(GlobalStoreStmt *stmt) override {
    DEFINE_BASIC_CHECK(GlobalStoreStmt)
    DEFINE_FIELD_CHECK(type_hint())
  }

  void visit(PragmaSLPStmt *stmt) override {
    DEFINE_BASIC_CHECK(PragmaSLPStmt)
    DEFINE_FIELD_CHECK(slp_width)
  }

  void visit(ElementShuffleStmt *stmt) override {
    DEFINE_BASIC_CHECK(ElementShuffleStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(pointer)
  }

  void visit(RangeAssumptionStmt *stmt) override {
    DEFINE_BASIC_CHECK(RangeAssumptionStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(low)
    DEFINE_FIELD_CHECK(high)
  }

  void visit(LinearizeStmt *stmt) override {
    DEFINE_BASIC_CHECK(LinearizeStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(inputs.size())
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      DEFINE_FIELD_CHECK(inputs[i])
    }
  }

  void visit(IntegerOffsetStmt *stmt) override {
    DEFINE_BASIC_CHECK(IntegerOffsetStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(offset)
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override {
    DEFINE_BASIC_CHECK(OffsetAndExtractBitsStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(offset)
    DEFINE_FIELD_CHECK(bit_begin)
    DEFINE_FIELD_CHECK(bit_end)
    DEFINE_FIELD_CHECK(simplified) // Is this necessary?
  }

  void visit(GetRootStmt *stmt) override {
    DEFINE_BASIC_CHECK(GetRootStmt)
    DEFINE_FIELD_CHECK(type_hint())
  }

  void visit(SNodeLookupStmt *stmt) override {
    DEFINE_BASIC_CHECK(SNodeLookupStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_SNODE_CHECK(snode)
    DEFINE_FIELD_CHECK(activate)
  }

  void visit(GetChStmt *stmt) override {
    DEFINE_BASIC_CHECK(GetChStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_SNODE_CHECK(input_snode)
    DEFINE_SNODE_CHECK(output_snode)
    DEFINE_FIELD_CHECK(chid)
  }

  void visit(ExternalPtrStmt *stmt) override {
    DEFINE_BASIC_CHECK(ExternalPtrStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(activate)
  }

  void visit(OffloadedStmt *stmt) override {
    DEFINE_BASIC_CHECK(OffloadedStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(task_type)
    DEFINE_SNODE_CHECK(snode)
    DEFINE_FIELD_CHECK(begin_offset)
    DEFINE_FIELD_CHECK(end_offset)
    DEFINE_FIELD_CHECK(const_begin)
    DEFINE_FIELD_CHECK(const_end)
    DEFINE_FIELD_CHECK(begin_value)
    DEFINE_FIELD_CHECK(end_value)
    // DEFINE_FIELD_CHECK(step) // seems never used
    DEFINE_FIELD_CHECK(block_dim)
    DEFINE_FIELD_CHECK(reversed)
    DEFINE_FIELD_CHECK(num_cpu_threads) // Is this necessary?
    DEFINE_FIELD_CHECK(device) // Is this necessary?
    if (stmt->has_body()) {
      TI_ASSERT(stmt->body);
      TI_ASSERT(other->body);
      other_node = other->body.get();
      stmt->body->accept(this);
      other_node = other;
    }
  }

  void visit(LoopIndexStmt *stmt) override {
    DEFINE_BASIC_CHECK(LoopIndexStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(index)
    DEFINE_FIELD_CHECK(is_struct_for)
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    DEFINE_BASIC_CHECK(GlobalTemporaryStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(offset)
  }

  void visit(InternalFuncStmt *stmt) override {
    DEFINE_BASIC_CHECK(InternalFuncStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(func_name)
  }

  void visit(StackAllocaStmt *stmt) override {
    DEFINE_BASIC_CHECK(StackAllocaStmt)
    DEFINE_FIELD_CHECK(type_hint())
    DEFINE_FIELD_CHECK(max_size)
  }

  void visit(StackLoadTopStmt *stmt) override {
    DEFINE_BASIC_CHECK(StackLoadTopStmt)
    DEFINE_FIELD_CHECK(type_hint())
  }

  void visit(StackLoadTopAdjStmt *stmt) override {
    DEFINE_BASIC_CHECK(StackLoadTopAdjStmt)
    DEFINE_FIELD_CHECK(type_hint())
  }

  void visit(StackPushStmt *stmt) override {
    DEFINE_BASIC_CHECK(StackPushStmt)
    DEFINE_FIELD_CHECK(type_hint())
  }

  void visit(StackPopStmt *stmt) override {
    DEFINE_BASIC_CHECK(StackPopStmt)
    DEFINE_FIELD_CHECK(type_hint())
  }

  void visit(StackAccAdjointStmt *stmt) override {
    DEFINE_BASIC_CHECK(StackAccAdjointStmt)
    DEFINE_FIELD_CHECK(type_hint())
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
