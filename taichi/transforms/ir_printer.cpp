// The IRPrinter prints the IR in a human-readable format

#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/frontend_ir.h"
#include <typeinfo>

TLANG_NAMESPACE_BEGIN

class IRPrinter : public IRVisitor {
 public:
  int current_indent;

  std::string *output;
  std::stringstream ss;

  IRPrinter(std::string *output = nullptr) : output(output) {
    current_indent = 0;
  }

  template <typename... Args>
  void print(std::string f, Args &&... args) {
    print_raw(fmt::format(f, std::forward<Args>(args)...));
  }

  void print_raw(std::string f) {
    for (int i = 0; i < current_indent; i++)
      f = "  " + f;
    f += "\n";
    if (output) {
      ss << f;
    } else {
      std::cout << f;
    }
  }

  static void run(IRNode *node, std::string *output) {
    auto p = IRPrinter(output);
    p.print("kernel {{");
    node->accept(&p);
    p.print("}}");
    if (output)
      *output = p.ss.str();
  }

  void visit(Block *stmt_list) override {
    current_indent++;
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    current_indent--;
  }

  void visit(FrontendBreakStmt *stmt) override {
    print("break");
  }

  void visit(FrontendContinueStmt *stmt) override {
    print("continue");
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
    std::string extras = "";
    for (auto &arg : assert->args) {
      extras += ", ";
      extras += arg->name();
    }
    print("{} : assert {}, \"{}\"{}", assert->id, assert->cond->name(),
          assert->text, extras);
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
          stmt->snode->get_node_type_name_hinted(), extras);
  }

  void visit(SNodeOpStmt *stmt) override {
    std::string extras;
    if (stmt->ptr)
      extras = "ptr = " + stmt->ptr->name();
    if (stmt->val) {
      extras += ", val = " + stmt->val->name();
    }
    if (!stmt->indices.empty()) {
      extras += " index [";
      for (int i = 0; i < (int)stmt->indices.size(); i++) {
        extras += fmt::format("{}", stmt->indices[i]->name());
        if (i + 1 < (int)stmt->indices.size()) {
          extras += ", ";
        }
      }
      extras += "]";
    }
    std::string snode = stmt->snode->get_node_type_name_hinted();
    print("{}{} = {} [{}] {}", stmt->type_hint(), stmt->name(),
          snode_op_type_name(stmt->op_type), snode, extras);
  }

  void visit(AllocaStmt *alloca) override {
    print("{}${} = alloca", alloca->type_hint(), alloca->id);
  }

  void visit(RandStmt *stmt) override {
    print("{}{} = rand()", stmt->type_hint(), stmt->name());
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->is_cast()) {
      std::string reint =
          stmt->op_type == UnaryOpType::cast_value ? "" : "reinterpret_";
      print("{}{} = {}{}<{}> {}", stmt->type_hint(), stmt->name(), reint,
            unary_op_type_name(stmt->op_type),
            data_type_short_name(stmt->cast_type), stmt->operand->name());
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

  void visit(AtomicOpStmt *stmt) override {
    print("{}{} = atomic {}({}, {})", stmt->type_hint(), stmt->name(),
          atomic_op_type_name(stmt->op_type), stmt->dest->name(),
          stmt->val->name());
  }

  void visit(IfStmt *if_stmt) override {
    print("{} : if {} {{", if_stmt->name(), if_stmt->cond->name());
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      print("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    print("}}");
  }

  void visit(FrontendIfStmt *if_stmt) override {
    print("{} : if {} {{", if_stmt->name(), if_stmt->condition->serialize());
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
    print("{} : while control {}, {}", stmt->name(),
          stmt->mask ? stmt->mask->name() : "nullptr", stmt->cond->name());
  }

  void visit(ContinueStmt *stmt) override {
    if (stmt->scope) {
      print("{} continue (scope={})", stmt->name(), stmt->name());
    } else {
      print("{} continue", stmt->name());
    }
  }

  void visit(FuncCallStmt *stmt) override {
    print("{}{} = call \"{}\"", stmt->type_hint(), stmt->name(), stmt->funcid);
  }

  void visit(FrontendFuncDefStmt *stmt) override {
    print("function \"{}\" {{", stmt->funcid);
    stmt->body->accept(this);
    print("}}");
  }

  void visit(FuncBodyStmt *stmt) override {
    print("func \"{}\" {{");
    stmt->body->accept(this);
    print("}}");
  }

  void visit(WhileStmt *stmt) override {
    print("{} : while true {{", stmt->name());
    stmt->body->accept(this);
    print("}}");
  }

  void visit(FrontendWhileStmt *stmt) override {
    print("{} : while {} {{", stmt->name(), stmt->cond->serialize());
    stmt->body->accept(this);
    print("}}");
  }

  void visit(FrontendForStmt *for_stmt) override {
    auto vars = make_list<Identifier>(
        for_stmt->loop_var_id,
        [](const Identifier &id) -> std::string { return id.name(); });
    if (for_stmt->is_ranged()) {
      print("{} : for {} in range({}, {}) {{", for_stmt->name(), vars,
            for_stmt->begin->serialize(), for_stmt->end->serialize());
    } else {
      print("{} : for {} where {} active {{", for_stmt->name(), vars,
            for_stmt->global_var.cast<GlobalVariableExpression>()
                ->snode->get_node_type_name_hinted());
    }
    for_stmt->body->accept(this);
    print("}}");
  }

  void visit(RangeForStmt *for_stmt) override {
    print("{} : {}for {} in range({}, {}, step {}) {{", for_stmt->name(),
          for_stmt->reversed ? "reversed " : "",
          for_stmt->loop_var ? for_stmt->loop_var->name() : "nullptr",
          for_stmt->begin->name(), for_stmt->end->name(), for_stmt->vectorize);
    for_stmt->body->accept(this);
    print("}}");
  }

  void visit(StructForStmt *for_stmt) override {
    auto loop_vars = make_list<Stmt *>(for_stmt->loop_vars,
                                       [](Stmt *const &stmt) -> std::string {
                                         return stmt ? stmt->name() : "nullptr";
                                       });
    print("{} : for {} where {} active, step {} {{", for_stmt->name(),
          loop_vars, for_stmt->snode->get_node_type_name_hinted(),
          for_stmt->vectorize);
    for_stmt->body->accept(this);
    print("}}");
  }

  void visit(GlobalPtrStmt *stmt) override {
    std::string s =
        fmt::format("{}{} = global ptr [", stmt->type_hint(), stmt->name());

    for (int l = 0; l < stmt->width(); l++) {
      std::string snode_name;
      if (stmt->snodes[l]) {
        snode_name = stmt->snodes[l]->get_node_type_name_hinted();
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

  void visit(FrontendKernelReturnStmt *stmt) override {
    print("{}{} : kernel return {}", stmt->type_hint(), stmt->name(),
          stmt->value->serialize());
  }

  void visit(KernelReturnStmt *stmt) override {
    print("{}{} : kernel return {}", stmt->type_hint(), stmt->name(),
          stmt->value->name());
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

    print("{}{} = linearized(ind {}, stride {})", stmt->type_hint(),
          stmt->name(), ind, stride);
  }

  void visit(IntegerOffsetStmt *stmt) override {
    print("{}{} = offset {} + {}", stmt->type_hint(), stmt->name(),
          stmt->input->name(), stmt->offset);
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override {
    print("{}{} = bit_extract({} + {}, {}~{})", stmt->type_hint(), stmt->name(),
          stmt->input->name(), stmt->offset, stmt->bit_begin, stmt->bit_end);
  }

  void visit(GetRootStmt *stmt) override {
    print("{}{} = get root", stmt->type_hint(), stmt->name());
  }

  void visit(SNodeLookupStmt *stmt) override {
    print("{}{} = [{}][{}]::lookup({}, {}) activate = {}", stmt->type_hint(),
          stmt->name(), stmt->snode->get_node_type_name_hinted(),
          stmt->snode->type_name(), stmt->input_snode->name(),
          stmt->input_index->name(), stmt->activate);
  }

  void visit(GetChStmt *stmt) override {
    print("{}{} = get child [{}->{}] {}", stmt->type_hint(), stmt->name(),
          stmt->input_snode->get_node_type_name_hinted(),
          stmt->output_snode->get_node_type_name_hinted(),
          stmt->input_ptr->name());
  }

  void visit(ExternalPtrStmt *stmt) override {
    std::string s = "<";
    for (int i = 0; i < (int)stmt->base_ptrs.size(); i++) {
      s += fmt::format("{}", stmt->base_ptrs[i]->name());
      if (i + 1 < (int)stmt->base_ptrs.size()) {
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

    print(fmt::format("{}{} = external_ptr {}", stmt->type_hint(), stmt->name(),
                      s));
  }

  void visit(OffloadedStmt *stmt) override {
    std::string details;
    if (stmt->task_type == stmt->range_for) {
      std::string begin_str, end_str;
      if (stmt->const_begin) {
        begin_str = std::to_string(stmt->begin_value);
      } else {
        begin_str = fmt::format("tmp(offset={}B)", stmt->begin_offset);
      }
      if (stmt->const_end) {
        end_str = std::to_string(stmt->end_value);
      } else {
        end_str = fmt::format("tmp(offset={}B)", stmt->end_offset);
      }
      details = fmt::format(
          "range_for({}, {}) block_dim={}", begin_str, end_str,
          stmt->block_dim == 0 ? "adaptive" : std::to_string(stmt->block_dim));
    } else if (stmt->task_type == stmt->struct_for) {
      details = fmt::format("struct_for({}) block_dim={}",
                            stmt->snode->get_node_type_name_hinted(),
                            stmt->block_dim);
    }
    if (stmt->task_type == OffloadedStmt::TaskType::listgen) {
      print("{} = offloaded listgen {}->{}", stmt->name(),
            stmt->snode->parent->get_node_type_name_hinted(),
            stmt->snode->get_node_type_name_hinted());
    } else if (stmt->task_type == OffloadedStmt::TaskType::clear_list) {
      print("{} = offloaded clear_list {}", stmt->name(),
            stmt->snode->get_node_type_name_hinted());
    } else if (stmt->task_type == OffloadedStmt::TaskType::gc) {
      print("{} = offloaded garbage collect {}", stmt->name(),
            stmt->snode->get_node_type_name_hinted());
    } else {
      print("{} = offloaded {} {{", stmt->name(), details);
      TI_ASSERT(stmt->body);
      stmt->body->accept(this);
      print("}}");
    }
  }

  void visit(LoopIndexStmt *stmt) override {
    print("{}{} = loop {} index {}", stmt->type_hint(), stmt->name(),
          stmt->loop->name(), stmt->index);
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    print("{}{} = global tmp var (offset = {} B)", stmt->type_hint(),
          stmt->name(), stmt->offset);
  }

  void visit(InternalFuncStmt *stmt) override {
    print("{} = call internal \"{}\"", stmt->name(), stmt->func_name);
  }

  void visit(StackAllocaStmt *stmt) override {
    print("{}{} = stack alloc (max_size={})", stmt->type_hint(), stmt->name(),
          stmt->max_size);
  }

  void visit(StackLoadTopStmt *stmt) override {
    print("{}{} = stack load top {}", stmt->type_hint(), stmt->name(),
          stmt->stack->name());
  }

  void visit(StackLoadTopAdjStmt *stmt) override {
    print("{}{} = stack load top adj {}", stmt->type_hint(), stmt->name(),
          stmt->stack->name());
  }

  void visit(StackPushStmt *stmt) override {
    print("{}{} = stack push {}, val = {}", stmt->type_hint(), stmt->name(),
          stmt->stack->name(), stmt->v->name());
  }

  void visit(StackPopStmt *stmt) override {
    print("{}{} : stack pop {}", stmt->type_hint(), stmt->name(),
          stmt->stack->name());
  }

  void visit(StackAccAdjointStmt *stmt) override {
    print("{}{} : stack acc adj {}, val = {}", stmt->type_hint(), stmt->name(),
          stmt->stack->name(), stmt->v->name());
  }
};

namespace irpass {

void print(IRNode *root, std::string *output) {
  return IRPrinter::run(root, output);
}

}  // namespace irpass

TLANG_NAMESPACE_END
