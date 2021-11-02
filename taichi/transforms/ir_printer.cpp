// The IRPrinter prints the IR in a human-readable format

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/util/str.h"

TLANG_NAMESPACE_BEGIN

namespace {

std::string scratch_pad_info(const MemoryAccessOptions &opt) {
  std::string ser;
  if (!opt.get_all().empty()) {
    ser += "mem_access_opt [ ";
    for (auto &rec : opt.get_all()) {
      for (auto flag : rec.second) {
        ser += rec.first->get_node_type_name_hinted() + ":" +
               snode_access_flag_name(flag) + " ";
      }
    }
    ser += "] ";
  } else {
    ser = "none";
  }
  return ser;
}

std::string block_dim_info(int block_dim) {
  return "block_dim=" +
         (block_dim == 0 ? "adaptive" : std::to_string(block_dim)) + " ";
}

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
      f = f.insert(0, "  ");
    f += "\n";
    if (output) {
      ss << f;
    } else {
      std::cout << f;
    }
  }

  static void run(IRNode *node, std::string *output) {
    if (node == nullptr) {
      TI_WARN("IRPrinter: Printing nullptr.");
      if (output) {
        *output = std::string();
      }
      return;
    }
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

  void visit(FrontendExprStmt *stmt) override {
    print("{}", stmt->val.serialize());
  }

  void visit(FrontendBreakStmt *stmt) override {
    print("break");
  }

  void visit(FrontendContinueStmt *stmt) override {
    print("continue");
  }

  void visit(FrontendAssignStmt *assign) override {
    print("{} = {}", assign->lhs.serialize(), assign->rhs.serialize());
  }

  void visit(FrontendAllocaStmt *alloca) override {
    print("{}${} = alloca {}", alloca->type_hint(), alloca->id,
          alloca->ident.name());
  }

  void visit(FrontendAssertStmt *assert) override {
    print("{} : assert {}", assert->id, assert->cond.serialize());
  }

  void visit(AssertStmt *assert) override {
    std::string extras;
    for (auto &arg : assert->args) {
      extras += ", ";
      extras += arg->name();
    }
    print("{} : assert {}, \"{}\"{}", assert->id, assert->cond->name(),
          assert->text, extras);
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    std::string extras = "inputs=";
    for (auto &arg : stmt->arg_stmts) {
      extras += ", ";
      extras += arg->name();
    }
    extras += "outputs=";
    for (auto &output : stmt->output_stmts) {
      extras += ", ";
      extras += output->name();
    }
    print("{} : func_call {:x}, {}", stmt->name(), (std::size_t)stmt->func,
          extras);
  }

  void visit(FrontendSNodeOpStmt *stmt) override {
    std::string extras = "[";
    for (int i = 0; i < (int)stmt->indices.size(); i++) {
      extras += stmt->indices[i].serialize();
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
    print("{} : if {} {{", if_stmt->name(), if_stmt->condition.serialize());
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      print("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    print("}}");
  }

  void visit(FrontendEvalStmt *stmt) override {
    print("{} = eval {}", stmt->name(), stmt->expr.serialize());
  }

  void visit(FrontendPrintStmt *print_stmt) override {
    std::vector<std::string> contents;
    for (auto const &c : print_stmt->contents) {
      std::string name;
      if (std::holds_alternative<Expr>(c))
        name = std::get<Expr>(c).serialize();
      else
        name = c_quoted(std::get<std::string>(c));
      contents.push_back(name);
    }
    print("print {}", fmt::join(contents, ", "));
  }

  void visit(PrintStmt *print_stmt) override {
    std::vector<std::string> names;
    for (auto const &c : print_stmt->contents) {
      std::string name;
      if (std::holds_alternative<Stmt *>(c))
        name = std::get<Stmt *>(c)->name();
      else
        name = c_quoted(std::get<std::string>(c));
      names.push_back(name);
    }
    print("print {}", fmt::join(names, ", "));
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
      print("{} continue (scope={})", stmt->name(), stmt->scope->name());
    } else {
      print("{} continue", stmt->name());
    }
  }

  void visit(FuncCallStmt *stmt) override {
    std::vector<std::string> args;
    for (const auto &arg : stmt->args) {
      args.push_back(arg->name());
    }
    print("{}{} = call \"{}\", args = {{{}}}", stmt->type_hint(), stmt->name(),
          stmt->func->get_name(), fmt::join(args, ", "));
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
    print("{} : while {} {{", stmt->name(), stmt->cond.serialize());
    stmt->body->accept(this);
    print("}}");
  }

  void visit(FrontendForStmt *for_stmt) override {
    auto vars = make_list<Identifier>(
        for_stmt->loop_var_id,
        [](const Identifier &id) -> std::string { return id.name(); });
    if (for_stmt->is_ranged()) {
      print("{} : for {} in range({}, {}) {}{{", for_stmt->name(), vars,
            for_stmt->begin.serialize(), for_stmt->end.serialize(),
            block_dim_info(for_stmt->block_dim));
    } else {
      print("{} : for {} in {} {}{}{{", for_stmt->name(), vars,
            for_stmt->global_var.is<GlobalVariableExpression>()
                ? for_stmt->global_var.cast<GlobalVariableExpression>()
                      ->snode->get_node_type_name_hinted()
                : for_stmt->global_var.serialize(),
            scratch_pad_info(for_stmt->mem_access_opt),
            block_dim_info(for_stmt->block_dim));
    }
    for_stmt->body->accept(this);
    print("}}");
  }

  void visit(RangeForStmt *for_stmt) override {
    print("{} : {}for in range({}, {}) (vectorize {}) (bit_vectorize {}) {}{{",
          for_stmt->name(), for_stmt->reversed ? "reversed " : "",
          for_stmt->begin->name(), for_stmt->end->name(), for_stmt->vectorize,
          for_stmt->bit_vectorize, block_dim_info(for_stmt->block_dim));
    for_stmt->body->accept(this);
    print("}}");
  }

  void visit(StructForStmt *for_stmt) override {
    print("{} : struct for in {} (vectorize {}) (bit_vectorize {}) {}{}{{",
          for_stmt->name(), for_stmt->snode->get_node_type_name_hinted(),
          for_stmt->vectorize, for_stmt->bit_vectorize,
          scratch_pad_info(for_stmt->mem_access_opt),
          block_dim_info(for_stmt->block_dim));
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

  void visit(PtrOffsetStmt *stmt) override {
    std::string s =
        fmt::format("{}{} = shift ptr [{} + {}]", stmt->type_hint(),
                    stmt->name(), stmt->origin->name(), stmt->offset->name());
    print_raw(s);
  }

  void visit(ArgLoadStmt *stmt) override {
    print("{}{} = arg[{}]", stmt->type_hint(), stmt->name(), stmt->arg_id);
  }

  void visit(FrontendReturnStmt *stmt) override {
    print("{}{} : return {}", stmt->type_hint(), stmt->name(),
          stmt->value.serialize());
  }

  void visit(ReturnStmt *stmt) override {
    print("{}{} : return {}", stmt->type_hint(), stmt->name(),
          stmt->value->name());
  }

  void visit(LocalLoadStmt *stmt) override {
    print("{}{} = local load [{}]", stmt->type_hint(), stmt->name(),
          to_string(stmt->src));
  }

  void visit(LocalStoreStmt *stmt) override {
    print("{}{} : local store [{} <- {}]", stmt->type_hint(), stmt->name(),
          stmt->dest->name(), stmt->val->name());
  }

  void visit(GlobalLoadStmt *stmt) override {
    print("{}{} = global load {}", stmt->type_hint(), stmt->name(),
          stmt->src->name());
  }

  void visit(GlobalStoreStmt *stmt) override {
    print("{}{} : global store [{} <- {}]", stmt->type_hint(), stmt->name(),
          stmt->dest->name(), stmt->val->name());
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

  void visit(LoopUniqueStmt *stmt) override {
    std::string add = "";
    if (!stmt->covers.empty()) {
      add = ", covers=[";
      for (const auto &sn : stmt->covers) {
        add += fmt::format("S{}, ", sn);
      }
      add.erase(add.size() - 2, 2);  // remove the last ", "
      add += "]";
    }
    print("{}{} = loop_unique({}{})", stmt->type_hint(), stmt->name(),
          stmt->input->name(), add);
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

  void visit(BitExtractStmt *stmt) override {
    print("{}{} = bit_extract({}) bit_range=[{}, {})", stmt->type_hint(),
          stmt->name(), stmt->input->name(), stmt->bit_begin, stmt->bit_end);
  }

  void visit(GetRootStmt *stmt) override {
    if (stmt->root() == nullptr)
      print("{}{} = get root nullptr", stmt->type_hint(), stmt->name());
    else
      print("{}{} = get root [{}][{}]", stmt->type_hint(), stmt->name(),
            stmt->root()->get_node_type_name_hinted(),
            stmt->root()->type_name());
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
    if (stmt->task_type == OffloadedTaskType::range_for) {
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
      details =
          fmt::format("range_for({}, {}) grid_dim={} block_dim={}", begin_str,
                      end_str, stmt->grid_dim, stmt->block_dim);
    } else if (stmt->task_type == OffloadedTaskType::struct_for) {
      details =
          fmt::format("struct_for({}) grid_dim={} block_dim={} bls={}",
                      stmt->snode->get_node_type_name_hinted(), stmt->grid_dim,
                      stmt->block_dim, scratch_pad_info(stmt->mem_access_opt));
    }
    if (stmt->task_type == OffloadedTaskType::listgen) {
      print("{} = offloaded listgen {}->{}", stmt->name(),
            stmt->snode->parent->get_node_type_name_hinted(),
            stmt->snode->get_node_type_name_hinted());
    } else if (stmt->task_type == OffloadedTaskType::gc) {
      print("{} = offloaded garbage collect {}", stmt->name(),
            stmt->snode->get_node_type_name_hinted());
    } else {
      print("{} = offloaded {} ", stmt->name(), details);
      if (stmt->tls_prologue) {
        print("tls prologue {{");
        stmt->tls_prologue->accept(this);
        print("}}");
      }
      if (stmt->bls_prologue) {
        print("bls prologue {{");
        stmt->bls_prologue->accept(this);
        print("}}");
      }
      TI_ASSERT(stmt->body);
      print("body {{");
      stmt->body->accept(this);
      print("}}");
      if (stmt->bls_epilogue) {
        print("bls_epilogue {{");
        stmt->bls_epilogue->accept(this);
        print("}}");
      }
      if (stmt->tls_epilogue) {
        print("tls_epilogue {{");
        stmt->tls_epilogue->accept(this);
        print("}}");
      }
    }
  }

  void visit(ClearListStmt *stmt) override {
    print("{} = clear_list {}", stmt->name(),
          stmt->snode->get_node_type_name_hinted());
  }

  void visit(LoopIndexStmt *stmt) override {
    print("{}{} = loop {} index {}", stmt->type_hint(), stmt->name(),
          stmt->loop->name(), stmt->index);
  }

  void visit(LoopLinearIndexStmt *stmt) override {
    print("{}{} = loop {} index linear", stmt->type_hint(), stmt->name(),
          stmt->loop->name());
  }

  void visit(BlockCornerIndexStmt *stmt) override {
    print("{}{} = loop {} block corner index {}", stmt->type_hint(),
          stmt->name(), stmt->loop->name(), stmt->index);
  }

  void visit(BlockDimStmt *stmt) override {
    print("{}{} = block dim", stmt->type_hint(), stmt->name());
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    print("{}{} = global tmp var (offset = {} B)", stmt->type_hint(),
          stmt->name(), stmt->offset);
  }

  void visit(ThreadLocalPtrStmt *stmt) override {
    print("{}{} = thread local ptr (offset = {} B)", stmt->type_hint(),
          stmt->name(), stmt->offset);
  }

  void visit(BlockLocalPtrStmt *stmt) override {
    print("{}{} = block local ptr (offset = {})", stmt->type_hint(),
          stmt->name(), stmt->offset->name());
  }

  void visit(InternalFuncStmt *stmt) override {
    std::string args;
    bool first = true;
    for (auto &arg : stmt->args) {
      if (!first) {
        args += ", ";
      }
      args += arg->name();
      first = false;
    }
    print("{}{} = internal call {}({})", stmt->type_hint(), stmt->name(),
          stmt->func_name, args);
  }

  void visit(AdStackAllocaStmt *stmt) override {
    print("{}{} = stack alloc (max_size={})", stmt->type_hint(), stmt->name(),
          stmt->max_size);
  }

  void visit(AdStackLoadTopStmt *stmt) override {
    print("{}{} = stack load top {}", stmt->type_hint(), stmt->name(),
          stmt->stack->name());
  }

  void visit(AdStackLoadTopAdjStmt *stmt) override {
    print("{}{} = stack load top adj {}", stmt->type_hint(), stmt->name(),
          stmt->stack->name());
  }

  void visit(AdStackPushStmt *stmt) override {
    print("{}{} : stack push {}, val = {}", stmt->type_hint(), stmt->name(),
          stmt->stack->name(), stmt->v->name());
  }

  void visit(AdStackPopStmt *stmt) override {
    print("{}{} : stack pop {}", stmt->type_hint(), stmt->name(),
          stmt->stack->name());
  }

  void visit(AdStackAccAdjointStmt *stmt) override {
    print("{}{} : stack acc adj {}, val = {}", stmt->type_hint(), stmt->name(),
          stmt->stack->name(), stmt->v->name());
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    print("{}{} = external_tensor_shape_along_axis {}, arg_id {}",
          stmt->type_hint(), stmt->name(), stmt->axis, stmt->arg_id);
  }

  void visit(BitStructStoreStmt *stmt) override {
    std::string ch_ids;
    std::string values;
    for (int i = 0; i < stmt->ch_ids.size(); i++) {
      ch_ids += fmt::format("{}", stmt->ch_ids[i]);
      values += fmt::format("{}", stmt->values[i]->name());
      if (i != stmt->ch_ids.size() - 1) {
        ch_ids += ", ";
        values += ", ";
      }
    }
    print("{} : {}bit_struct_store {}, ch_ids=[{}], values=[{}]", stmt->name(),
          stmt->is_atomic ? "atomic " : "", stmt->ptr->name(), ch_ids, values);
  }
};

}  // namespace

namespace irpass {

void print(IRNode *root, std::string *output) {
  return IRPrinter::run(root, output);
}

}  // namespace irpass

TLANG_NAMESPACE_END
