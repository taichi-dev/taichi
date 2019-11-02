// The IRPrinter prints the IR in a human-readable format

#include <typeinfo>
#include "../ir.h"

TLANG_NAMESPACE_BEGIN

class IRPrinter : public IRVisitor {
 public:
  int current_indent;

  IRPrinter() {
    current_indent = 0;
  }

  template <typename... Args>
  void print(std::string f, Args &&... args) {
    print_raw(fmt::format(f, std::forward<Args>(args)...));
  }

  void print_raw(std::string f) {
    for (int i = 0; i < current_indent; i++)
      fmt::print("  ");
    std::cout << f;
    fmt::print("\n");
  }

  static void run(IRNode *node) {
    auto p = IRPrinter();
    fmt::print("==========\n");
    fmt::print("kernel {{\n");
    node->accept(&p);
    fmt::print("}}\n");
    fmt::print("==========\n");
  }

  void visit(Block *stmt_list) override {
    current_indent++;
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    current_indent--;
  }

  void visit(FrontendAssignStmt *assign) override {
    print("{} = {}", assign->lhs->serialize(), assign->rhs->serialize());
  }

  void visit(FrontendAllocaStmt *alloca) override {
    print("{}${} = alloca {}", alloca->type_hint(), alloca->id,
          alloca->ident.name());
  }

  void visit(FrontendAssertStmt *assert) override {
    print("{} : assert {}", assert->id, assert->val->serialize());
  }

  void visit(AssertStmt *assert) override {
    print("{} : assert {}", assert->id, assert->val->name());
  }

  void visit(FrontendSNodeOpStmt *stmt) override {
    std::string extras = "[";
    for (int i = 0; i < (int)stmt->indices.size(); i++) {
      extras += stmt->indices[i]->serialize();
      if (i + 1 < (int)stmt->indices.size())
        extras += ", ";
    }
    extras += "]";
    if (stmt->val.expr) {
      extras += ", " + stmt->val.serialize();
    }
    print("{} : {} {} {}", stmt->name(), snode_op_type_name(stmt->op_type),
          stmt->snode->node_type_name, extras);
  }

  void visit(SNodeOpStmt *stmt) override {
    std::string extras = "[";
    for (int i = 0; i < (int)stmt->indices.size(); i++) {
      extras += stmt->indices[i]->name();
      if (i + 1 < (int)stmt->indices.size())
        extras += ", ";
    }
    extras += "]";
    if (stmt->val) {
      extras += ", " + stmt->val->name();
    }
    std::string snodes;
    for (int l = 0; l < stmt->width(); l++) {
      snodes += stmt->snodes[l]->node_type_name;
      if (l > 0)
        snodes += ", ";
    }
    print("{} : {} [{}] {}", stmt->name(), snode_op_type_name(stmt->op_type),
          snodes, extras);
  }

  void visit(AllocaStmt *alloca) override {
    print("{}${} = alloca", alloca->type_hint(), alloca->id);
  }

  void visit(RandStmt *stmt) override {
    print("{}{} = rand()", stmt->type_hint(), stmt->name());
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->op_type == UnaryOpType::cast) {
      std::string reint = stmt->cast_by_value ? "" : "reinterpret_";
      print("{}{} = {}{}<{}> {}", stmt->type_hint(), stmt->name(), reint,
            unary_op_type_name(stmt->op_type), data_type_name(stmt->cast_type),
            stmt->operand->name());
    } else {
      print("{}{} = {} {}", stmt->type_hint(), stmt->name(),
            unary_op_type_name(stmt->op_type), stmt->operand->name());
    }
  }

  void visit(BinaryOpStmt *bin) override {
    print("{}{} = {} {} {}", bin->type_hint(), bin->name(),
          binary_op_type_name(bin->op_type), bin->lhs->name(),
          bin->rhs->name());
  }

  void visit(TernaryOpStmt *stmt) override {
    print("{}{} = {}({}, {}, {})", stmt->type_hint(), stmt->name(),
          ternary_type_name(stmt->op_type), stmt->op1->name(),
          stmt->op2->name(), stmt->op3->name());
  }

  void visit(FrontendAtomicStmt *stmt) override {
    print("{}{} = atomic {}({}, {})", stmt->type_hint(), stmt->name(),
          atomic_op_type_name(stmt->op_type), stmt->dest->serialize(),
          stmt->val->serialize());
  }

  void visit(AtomicOpStmt *stmt) override {
    print("{}{} = atomic {}({}, {})", stmt->type_hint(), stmt->name(),
          atomic_op_type_name(stmt->op_type), stmt->dest->name(),
          stmt->val->name());
  }

  void visit(IfStmt *if_stmt) override {
    print("if {} {{", if_stmt->cond->name());
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      print("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    print("}}");
  }

  void visit(FrontendIfStmt *if_stmt) override {
    print("if {} {{", if_stmt->condition->serialize());
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      print("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    print("}}");
  }

  void visit(FrontendPrintStmt *print_stmt) override {
    print("print \"{}\", {}", print_stmt->str, print_stmt->expr.serialize());
  }

  void visit(FrontendEvalStmt *stmt) override {
    print("{} = eval {}", stmt->name(), stmt->expr.serialize());
  }

  void visit(PrintStmt *print_stmt) override {
    print("{}print {}, {}", print_stmt->type_hint(), print_stmt->str,
          print_stmt->stmt->name());
  }

  void visit(ConstStmt *const_stmt) override {
    print("{}{} = const {}", const_stmt->type_hint(), const_stmt->name(),
          const_stmt->val.serialize(
              [](const TypedConstant &t) { return t.stringify(); }, "["));
  }

  void visit(WhileControlStmt *stmt) override {
    print("while control {}, {}", stmt->mask->name(), stmt->cond->name());
  }

  void visit(WhileStmt *stmt) override {
    print("while true {{");
    stmt->body->accept(this);
    print("}}");
  }

  void visit(FrontendWhileStmt *stmt) override {
    print("while {} {{", stmt->cond->serialize());
    stmt->body->accept(this);
    print("}}");
  }

  void visit(FrontendForStmt *for_stmt) override {
    auto vars = make_list<Ident>(
        for_stmt->loop_var_id,
        [](const Ident &id) -> std::string { return id.name(); });
    if (for_stmt->is_ranged()) {
      print("for {} in range({}, {}) {{", vars, for_stmt->begin->serialize(),
            for_stmt->end->serialize());
    } else {
      print("for {} where {} active {{", vars,
            for_stmt->global_var.cast<GlobalVariableExpression>()
                ->snode->node_type_name);
    }
    for_stmt->body->accept(this);
    print("}}");
  }

  void visit(RangeForStmt *for_stmt) override {
    print("{}for {} in range({}, {}, step {}) {{",
          for_stmt->reversed ? "reversed " : "", for_stmt->loop_var->name(),
          for_stmt->begin->name(), for_stmt->end->name(), for_stmt->vectorize);
    for_stmt->body->accept(this);
    print("}}");
  }

  void visit(StructForStmt *for_stmt) override {
    auto loop_vars = make_list<Stmt *>(
        for_stmt->loop_vars,
        [](Stmt *const &stmt) -> std::string { return stmt->name(); });
    print("for {} where {} active, step {} {{", loop_vars,
          for_stmt->snode->node_type_name, for_stmt->vectorize);
    for_stmt->body->accept(this);
    print("}}");
  }

  void visit(GlobalPtrStmt *stmt) override {
    std::string s =
        fmt::format("{}{} = global ptr [", stmt->type_hint(), stmt->name());

    for (int l = 0; l < stmt->width(); l++) {
      std::string snode_name;
      if (stmt->snodes[l]) {
        snode_name = stmt->snodes[l]->node_type_name;
      } else {
        snode_name = "unknown";
      }
      s += snode_name;
      if (l + 1 < stmt->width()) {
        s += ", ";
      }
    }
    s += "], index [";
    for (int i = 0; i < (int)stmt->indices.size(); i++) {
      s += fmt::format("{}", stmt->indices[i]->name());
      if (i + 1 < (int)stmt->indices.size()) {
        s += ", ";
      }
    }
    s += "]";

    s += " activate=" + std::string(stmt->activate ? "true" : "false");

    print_raw(s);
  }

  void visit(ArgLoadStmt *stmt) override {
    print("{}{} = arg[{}]", stmt->type_hint(), stmt->name(), stmt->arg_id);
  }

  void visit(FrontendArgStoreStmt *stmt) override {
    print("{}{} : store arg {} <- {}", stmt->type_hint(), stmt->name(),
          stmt->arg_id, stmt->expr->serialize());
  }

  void visit(ArgStoreStmt *stmt) override {
    print("{}{} : store arg {} <- {}", stmt->type_hint(), stmt->name(),
          stmt->arg_id, stmt->val->name());
  }

  void visit(LocalLoadStmt *stmt) override {
    print("{}{} = local load [{}]", stmt->type_hint(), stmt->name(),
          to_string(stmt->ptr));
  }

  void visit(LocalStoreStmt *stmt) override {
    print("{}{} : local store [{} <- {}]", stmt->type_hint(), stmt->name(),
          stmt->ptr->name(), stmt->data->name());
  }

  void visit(GlobalLoadStmt *stmt) override {
    print("{}{} = global load {}", stmt->type_hint(), stmt->name(),
          stmt->ptr->name());
  }

  void visit(GlobalStoreStmt *stmt) override {
    print("{}{} : global store [{} <- {}]", stmt->type_hint(), stmt->name(),
          stmt->ptr->name(), stmt->data->name());
  }

  void visit(PragmaSLPStmt *stmt) override {
    print("#pragma SLP({})", stmt->slp_width);
  }

  void visit(ElementShuffleStmt *stmt) override {
    print("{}{} = shuffle {}", stmt->type_hint(), stmt->name(),
          stmt->elements.serialize([](const VectorElement &ve) {
            return fmt::format("{}[{}]", ve.stmt->name(), ve.index);
          }));
  }

  void visit(RangeAssumptionStmt *stmt) override {
    print("{}{} = assume_in_range({}{:+d} <= {} < {}{:+d})", stmt->type_hint(),
          stmt->name(), stmt->base->name(), stmt->low, stmt->input->name(),
          stmt->base->name(), stmt->high);
  }

  void visit(LinearizeStmt *stmt) override {
    auto ind = make_list<Stmt *>(
        stmt->inputs, [&](Stmt *const &stmt) { return stmt->name(); }, "{");
    auto stride = make_list<int>(
        stmt->strides,
        [&](const int &stride) { return std::to_string(stride); }, "{");

    print("{} = linearized(ind {}, stride {})", stmt->name(), ind, stride);
  }

  void visit(IntegerOffsetStmt *stmt) override {
    print("{} = offset {} + {}", stmt->name(), stmt->input->name(),
          stmt->offset);
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override {
    print("{} = bit_extract({} + {}, {}~{})", stmt->name(), stmt->input->name(),
          stmt->offset, stmt->bit_begin, stmt->bit_end);
  }
  void visit(SNodeLookupStmt *stmt) override {
    print(
        "{} = [{}][{}]::lookup({}, {}) coord = {} activate = {}", stmt->name(),
        stmt->snode->node_type_name, stmt->snode->type_name(),
        stmt->input_snode ? stmt->input_snode->name() : "root",
        stmt->input_index->name(),
        make_list<Stmt *>(stmt->global_indices,
                          [&](Stmt *const &stmt) { return stmt->name(); }, "{"),
        stmt->activate);
  }

  void visit(GetChStmt *stmt) override {
    print("{} = get child [{}->{}] {}", stmt->name(),
          stmt->input_snode->node_type_name, stmt->output_snode->node_type_name,
          stmt->input_ptr->name());
  }

  void visit(ClearAllStmt *stmt) override {
    print("{} = clear {} deactivate={}", stmt->name(),
          stmt->snode->node_type_name, stmt->deactivate);
  }

  void visit(ExternalPtrStmt *stmt) override {
    std::string s = "<";
    for (int i = 0; i < (int)stmt->indices.size(); i++) {
      s += fmt::format("{}", stmt->base_ptrs[i]->name());
      if (i + 1 < (int)stmt->indices.size()) {
        s += ", ";
      }
    }
    s += ">, [";
    for (int i = 0; i < (int)stmt->indices.size(); i++) {
      s += fmt::format("{}", stmt->indices[i]->name());
      if (i + 1 < (int)stmt->indices.size()) {
        s += ", ";
      }
    }
    s += "]";

    print(fmt::format("{} = external_ptr {}", stmt->name(), s));
  }

  void visit(OffloadedStmt *stmt) override {
    std::string details;
    if (stmt->task_type == stmt->range_for) {
      details = fmt::format(" range_for({}, {})", stmt->begin, stmt->end);
    } else if (stmt->task_type == stmt->struct_for) {
      details =
          fmt::format(" struct_for({})", stmt->snode->get_node_type_name());
    }
    if (stmt->task_type == OffloadedStmt::TaskType::listgen) {
      print("{} = offloaded listgen {}", stmt->name(),
            stmt->snode->get_node_type_name());
    } else {
      print("{} = offloaded {} {{", stmt->name(), details);
      TC_ASSERT(stmt->body);
      stmt->body->accept(this);
      print("}}");
    }
  }

  void visit(LoopIndexStmt *stmt) override {
    print("{} = loop index {}", stmt->name(), stmt->index);
  }
};

namespace irpass {

void print(IRNode *root) {
  return IRPrinter::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
