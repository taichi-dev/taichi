#include "codegen_cc.h"
#include "cc_kernel.h"
#include "cc_layout.h"
#include "cc_program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/util/line_appender.h"
#include "taichi/util/str.h"
#include "taichi/llvm/llvm_program.h"
#include "cc_utils.h"

#define C90_COMPAT 0

TLANG_NAMESPACE_BEGIN
namespace cccp {  // Codegen for C Compiler Processor

namespace {
std::string get_node_ptr_name(SNode *snode) {
  return fmt::format("struct Ti_{} *", snode->get_node_type_name_hinted());
}
}  // namespace

class CCTransformer : public IRVisitor {
 private:
  [[maybe_unused]] Kernel *kernel;
  [[maybe_unused]] CCLayout *layout;

  LineAppender line_appender;
  LineAppender line_appender_header;
  bool is_top_level{true};
  GetRootStmt *root_stmt;

 public:
  CCTransformer(Kernel *kernel, CCLayout *layout)
      : kernel(kernel), layout(layout) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void run() {
    this->lower_ast();
    emit_header("void Tk_{}(struct Ti_Context *ti_ctx) {{", kernel->name);
    kernel->ir->accept(this);
    emit("}}");
  }

  void lower_ast() {
    auto ir = kernel->ir.get();
    auto config = kernel->program->config;
    config.demote_dense_struct_fors = true;
    irpass::compile_to_executable(ir, config, kernel,
                                  /*vectorize=*/false, kernel->grad,
                                  /*ad_use_stack=*/true, config.print_ir,
                                  /*lower_global_access*/ true);
  }

  std::string get_source() {
    return line_appender_header.lines() + line_appender.lines();
  }

 private:
  void visit(Block *stmt) override {
    if (!is_top_level)
      line_appender.push_indent();
    for (auto &s : stmt->statements) {
      s->accept(this);
    }
    if (!is_top_level)
      line_appender.pop_indent();
  }

  void visit(Stmt *stmt) override {
    TI_WARN("[cc] unsupported statement type {}", typeid(*stmt).name());
  }

  void visit(BitExtractStmt *stmt) override {
    emit("{} = (({} >> {}) & ((1 << {}) - 1));",
         define_var("Ti_i32", stmt->raw_name()), stmt->input->raw_name(),
         stmt->bit_begin, stmt->bit_end - stmt->bit_begin);
  }

  std::string define_var(std::string const &type, std::string const &name) {
    if (C90_COMPAT) {
      emit_header("{} {};", type, name);
      return name;
    } else {
      return fmt::format("{} {}", type, name);
    }
  }

  void visit(GetRootStmt *stmt) override {
    auto *root = kernel->program->get_snode_root(SNodeTree::kFirstID);
    emit("{} = ti_ctx->root;",
         define_var(get_node_ptr_name(root), stmt->raw_name()));
    root_stmt = stmt;
  }

  void visit(SNodeLookupStmt *stmt) override {
    Stmt *input_ptr;
    if (stmt->input_snode) {
      input_ptr = stmt->input_snode;
    } else {
      TI_ASSERT(root_stmt != nullptr);
      input_ptr = root_stmt;
    }

    emit("{} = &{}[{}];",
         define_var(get_node_ptr_name(stmt->snode), stmt->raw_name()),
         input_ptr->raw_name(), stmt->input_index->raw_name());
  }

  void visit(GetChStmt *stmt) override {
    auto snode = stmt->output_snode;
    std::string type;
    if (snode->type == SNodeType::place) {
      auto dt = fmt::format("{} *", cc_data_type_name(snode->dt));
      emit("{} = &{}->{};", define_var(dt, stmt->raw_name()),
           stmt->input_ptr->raw_name(), snode->get_node_type_name());
    } else {
      emit("{} = {}->{};",
           define_var(get_node_ptr_name(snode), stmt->raw_name()),
           stmt->input_ptr->raw_name(), snode->get_node_type_name());
    }
  }

  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("{} = *{};",
         define_var(cc_data_type_name(stmt->element_type()), stmt->raw_name()),
         stmt->src->raw_name());
  }

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("*{} = {};", stmt->dest->raw_name(), stmt->val->raw_name());
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto ptr_type =
        cc_data_type_name(stmt->element_type().ptr_removed()) + " *";
    auto var = define_var(ptr_type, stmt->raw_name());
    emit("{} = ({}) (ti_ctx->gtmp + {});", var, ptr_type, stmt->offset);
  }

  void visit(LinearizeStmt *stmt) override {
    std::string val = "0";
    for (int i = 0; i < stmt->inputs.size(); i++) {
      val = fmt::format("({} * {} + {})", val, stmt->strides[i],
                        stmt->inputs[i]->raw_name());
    }
    emit("{} = {};", define_var("Ti_i32", stmt->raw_name()), val);
  }

  void visit(ExternalPtrStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    std::string offset = "0";
    const auto *argload = stmt->base_ptrs[0]->as<ArgLoadStmt>();
    const int arg_id = argload->arg_id;
    for (int i = 0; i < stmt->indices.size(); i++) {
      auto stride = fmt::format("ti_ctx->earg[{} * {} + {}]", arg_id,
                                taichi_max_num_indices, i);
      offset = fmt::format("({} * {} + {})", offset, stride,
                           stmt->indices[i]->raw_name());
    }
    auto var =
        define_var(cc_data_type_name(stmt->element_type().ptr_removed()) + " *",
                   stmt->raw_name());
    emit("{} = {} + {};", var, stmt->base_ptrs[0]->raw_name(), offset);
  }

  void visit(ArgLoadStmt *stmt) override {
    if (stmt->is_ptr) {
      auto var = define_var(
          cc_data_type_name(stmt->element_type().ptr_removed()) + " *",
          stmt->raw_name());
      emit("{} = ti_ctx->args[{}].ptr_{};", var, stmt->arg_id,
           data_type_name(stmt->element_type().ptr_removed()));
    } else {
      auto var =
          define_var(cc_data_type_name(stmt->element_type()), stmt->raw_name());
      emit("{} = ti_ctx->args[{}].val_{};", var, stmt->arg_id,
           data_type_name(stmt->element_type()));
    }
  }

  void visit(ReturnStmt *stmt) override {
    emit("ti_ctx->args[0].val_{} = {};", data_type_name(stmt->element_type()),
         stmt->value->raw_name());
  }

  void visit(ConstStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("{} = {};",
         define_var(cc_data_type_name(stmt->element_type()), stmt->raw_name()),
         stmt->val[0].stringify());
  }

  void visit(AllocaStmt *stmt) override {
    emit("{} = 0;",
         define_var(cc_data_type_name(stmt->element_type()), stmt->raw_name()));
  }

  void visit(LocalLoadStmt *stmt) override {
    bool linear_index = true;
    for (int i = 0; i < (int)stmt->src.size(); i++) {
      if (stmt->src[i].offset != i) {
        linear_index = false;
      }
    }
    TI_ASSERT(stmt->same_source() && linear_index &&
              stmt->width() == stmt->src[0].var->width());

    auto var =
        define_var(cc_data_type_name(stmt->element_type()), stmt->raw_name());
    emit("{} = {};", var, stmt->src[0].var->raw_name());
  }

  void visit(LocalStoreStmt *stmt) override {
    emit("{} = {};", stmt->dest->raw_name(), stmt->val->raw_name());
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    TI_ASSERT(!stmt->func);
    auto format = stmt->source;
    std::string source;

    for (int i = 0; i < format.size(); i++) {
      char c = format[i];
      if (c == '%' || c == '$') {  // '$' for output, '%' for input
        int num = 0;
        while (i < format.size()) {
          i += 1;
          if (!::isdigit(format[i])) {
            i -= 1;
            break;
          }
          num *= 10;
          num += format[i] - '0';
        }
        auto args = (c == '%') ? stmt->arg_stmts : stmt->output_stmts;
        TI_ASSERT_INFO(num < args.size(), "{}{} out of {} argument range {}", c,
                       num, ((c == '%') ? "input" : "output"), args.size());
        source += args[num]->raw_name();
      } else {
        source.push_back(c);
      }
    }

    emit("{};", source);
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    const auto type = cc_data_type_name(stmt->element_type());
    const auto name = stmt->raw_name();
    const auto var = define_var(type, name);
    const auto arg_id = stmt->arg_id;
    const auto axis = stmt->axis;
    const auto axis_size = fmt::format("ti_ctx->earg[{} * {} + {}]", arg_id,
                                       taichi_max_num_indices, axis);
    emit("{} = {};", var, axis_size);
  }

  static std::string get_libc_function_name(std::string name, DataType dt) {
    std::string ret;
    if (dt->is_primitive(PrimitiveTypeID::i32))
      ret = name;
    else if (dt->is_primitive(PrimitiveTypeID::i64))
      ret = "ll" + name;
    else if (dt->is_primitive(PrimitiveTypeID::f32))
      ret = name + "f";
    else if (dt->is_primitive(PrimitiveTypeID::f64))
      ret = name;
    else
      TI_ERROR("Unsupported function \"{}\" for DataType={} on C backend", name,
               data_type_name(dt));

    if (name == "rsqrt") {
      ret = "Ti_" + ret;
    } else if (name == "sgn") {
      if (is_real(dt)) {
        ret = "f" + ret;
      }
      ret = "Ti_" + ret;
    } else if (name == "max" || name == "min" || name == "abs") {
      if (is_real(dt)) {
        ret = "f" + ret;
      } else if (ret != "abs") {
        ret = "Ti_" + ret;
      }
    }
    return ret;
  }

  static std::string invoke_libc(std::string name,
                                 DataType dt,
                                 std::string arguments) {
    auto func_name = get_libc_function_name(name, dt);
    return fmt::format("{}({})", func_name, arguments);
  }

  template <typename... Args>
  static inline std::string invoke_libc(std::string name,
                                        DataType dt,
                                        std::string const &fmt,
                                        Args &&... args) {
    auto arguments = fmt::format(fmt, std::forward<Args>(args)...);
    return invoke_libc(name, dt, arguments);
  }

  void visit(TernaryOpStmt *tri) override {
    TI_ASSERT(tri->op_type == TernaryOpType::select);
    emit("{} {} = {} != 0 ? {} : {};", cc_data_type_name(tri->element_type()),
         tri->raw_name(), tri->op1->raw_name(), tri->op2->raw_name(),
         tri->op3->raw_name());
  }

  void visit(BinaryOpStmt *bin) override {
    TI_ASSERT(bin->width() == 1);
    const auto dt_name = cc_data_type_name(bin->element_type());
    const auto lhs_name = bin->lhs->raw_name();
    const auto rhs_name = bin->rhs->raw_name();
    const auto bin_name = bin->raw_name();
    const auto type = bin->element_type();
    const auto binop = binary_op_type_symbol(bin->op_type);
    const auto var = define_var(dt_name, bin_name);
    if (cc_is_binary_op_infix(bin->op_type)) {
      if (is_comparison(bin->op_type)) {
        // XXX(#577): Taichi uses -1 as true due to LLVM i1...
        emit("{} = -({} {} {});", var, lhs_name, binop, rhs_name);
      } else if (bin->op_type == BinaryOpType::truediv) {
        emit("{} = ({}) {} / {};", var, dt_name, lhs_name, rhs_name);
      } else if (bin->op_type == BinaryOpType::floordiv) {
        auto lhs_dt_name = data_type_name(bin->lhs->element_type());
        if (is_integral(bin->lhs->element_type()) &&
            is_integral(bin->rhs->element_type())) {
          emit("{} = Ti_floordiv_{}({}, {});", var, lhs_dt_name, lhs_name,
               rhs_name);
        } else {
          emit("{} = Ti_floordiv_{}({}, {});", var, lhs_dt_name, lhs_name,
               rhs_name);
        }
      } else {
        emit("{} = {} {} {};", var, lhs_name, binop, rhs_name);
      }
    } else {
      emit("{} = {};", var,
           invoke_libc(binop, type, "{}, {}", lhs_name, rhs_name));
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    const auto dt_name = cc_data_type_name(stmt->element_type());
    const auto operand_name = stmt->operand->raw_name();
    const auto dest_name = stmt->raw_name();
    const auto type = stmt->element_type();
    const auto op = unary_op_type_symbol(stmt->op_type);
    const auto var = define_var(dt_name, dest_name);
    if (stmt->op_type == UnaryOpType::cast_value) {
      emit("{} = ({}) {};", var, dt_name, operand_name);

    } else if (stmt->op_type == UnaryOpType::cast_bits) {
      const auto operand_dt_name =
          cc_data_type_name(stmt->operand->element_type());
      emit("union {{ {} bc_src; {} bc_dst; }} {}_bitcast;", operand_dt_name,
           dt_name, dest_name);
      emit("{}_bitcast.bc_src = {};", dest_name, operand_name);
      emit("{} = {}_bitcast.bc_dst;", var, dest_name);

    } else if (cc_is_unary_op_infix(stmt->op_type)) {
      emit("{} = {}{};", var, op, operand_name);
    } else {
      emit("{} = {};", var, invoke_libc(op, type, "{}", operand_name));
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    const auto dest_ptr = stmt->dest->raw_name();
    const auto src_name = stmt->val->raw_name();
    const auto op = cc_atomic_op_type_symbol(stmt->op_type);
    const auto type = stmt->dest->element_type().ptr_removed();
    auto var = define_var(cc_data_type_name(type), stmt->raw_name());
    emit("{} = *{};", var, dest_ptr);
    if (stmt->op_type == AtomicOpType::max ||
        stmt->op_type == AtomicOpType::min) {
      emit("*{} = {};", dest_ptr,
           invoke_libc(op, type, "*{}, {}", dest_ptr, src_name));
    } else {
      emit("*{} {}= {};", dest_ptr, op, src_name);
    }
  }

  void visit(PrintStmt *stmt) override {
    std::string format;
    std::vector<std::string> values;

    for (int i = 0; i < stmt->contents.size(); i++) {
      auto const &content = stmt->contents[i];

      if (std::holds_alternative<Stmt *>(content)) {
        auto arg_stmt = std::get<Stmt *>(content);
        format += data_type_format(arg_stmt->ret_type);
        values.push_back(arg_stmt->raw_name());

      } else {
        auto str = std::get<std::string>(content);
        format += "%s";
        values.push_back(c_quoted(str));
      }
    }

    values.insert(values.begin(), c_quoted(format));
    emit("printf({});", fmt::join(values, ", "));
  }

  void generate_serial_kernel(OffloadedStmt *stmt) {
    stmt->body->accept(this);
  }

  void generate_range_for_kernel(OffloadedStmt *stmt) {
    if (stmt->const_begin && stmt->const_end) {
      ScopedIndent _s(line_appender);
      auto begin_value = stmt->begin_value;
      auto end_value = stmt->end_value;
      auto var = define_var("Ti_i32", stmt->raw_name());
      emit("for ({} = {}; {} < {}; {} += {}) {{", var, begin_value,
           stmt->raw_name(), end_value, stmt->raw_name(), 1 /* stmt->step? */);
      stmt->body->accept(this);
      emit("}}");
    } else {
      auto var = define_var("Ti_i32", stmt->raw_name());
      auto begin_expr = "tmp_begin_" + stmt->raw_name();
      auto end_expr = "tmp_end_" + stmt->raw_name();
      auto begin_var = define_var("Ti_i32", begin_expr);
      auto end_var = define_var("Ti_i32", end_expr);
      if (!stmt->const_begin) {
        emit("{} = *(Ti_i32 *) (ti_ctx->gtmp + {});", begin_var,
             stmt->begin_offset);
      } else {
        emit("{} = {};", begin_var, stmt->begin_value);
      }
      if (!stmt->const_end) {
        emit("{} = *(Ti_i32 *) (ti_ctx->gtmp + {});", end_var,
             stmt->end_offset);
      } else {
        emit("{} = {};", end_var, stmt->end_value);
      }
      emit("for ({} = {}; {} < {}; {} += {}) {{", var, begin_expr,
           stmt->raw_name(), end_expr, stmt->raw_name(), 1 /* stmt->step? */);
      stmt->body->accept(this);
      emit("}}");
    }
  }

  void visit(OffloadedStmt *stmt) override {
    TI_ASSERT(is_top_level);
    is_top_level = false;
    if (stmt->task_type == OffloadedStmt::TaskType::serial) {
      generate_serial_kernel(stmt);
    } else if (stmt->task_type == OffloadedStmt::TaskType::range_for) {
      generate_range_for_kernel(stmt);
    } else {
      TI_ERROR("[glsl] Unsupported offload type={} on C backend",
               stmt->task_name());
    }
    is_top_level = true;
  }

  void visit(LoopIndexStmt *stmt) override {
    TI_ASSERT(stmt->index == 0);  // TODO: multiple indices
    if (stmt->loop->is<OffloadedStmt>()) {
      auto type = stmt->loop->as<OffloadedStmt>()->task_type;
      if (type == OffloadedStmt::TaskType::range_for) {
        emit("Ti_i32 {} = {};", stmt->raw_name(), stmt->loop->raw_name());
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->loop->is<RangeForStmt>()) {
      emit("Ti_i32 {} = {};", stmt->raw_name(), stmt->loop->raw_name());
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(RangeForStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto var = define_var("Ti_i32", stmt->raw_name());
    if (!stmt->reversed) {
      emit("for ({} = {}; {} < {}; {} += {}) {{", var, stmt->begin->raw_name(),
           stmt->raw_name(), stmt->end->raw_name(), stmt->raw_name(), 1);
    } else {
      // reversed for loop
      emit("for ({} = {} - {}; {} >= {}; {} -= {}) {{", var,
           stmt->end->raw_name(), 1, stmt->raw_name(), stmt->begin->raw_name(),
           stmt->raw_name(), 1);
    }
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(WhileControlStmt *stmt) override {
    emit("if (!{}) break;", stmt->cond->raw_name());
  }

  void visit(ContinueStmt *stmt) override {
    emit("continue;");
  }

  void visit(WhileStmt *stmt) override {
    emit("while (1) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(IfStmt *stmt) override {
    emit("if ({}) {{", stmt->cond->raw_name());
    if (stmt->true_statements) {
      stmt->true_statements->accept(this);
    }
    if (stmt->false_statements) {
      emit("}} else {{");
      stmt->false_statements->accept(this);
    }
    emit("}}");
  }

  void visit(RandStmt *stmt) override {
    auto var = define_var(cc_data_type_name(stmt->ret_type), stmt->raw_name());
    emit("{} = Ti_rand_{}();", var, data_type_name(stmt->ret_type));
  }

  void visit(AdStackAllocaStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    TI_ASSERT_INFO(
        stmt->max_size > 0,
        "Adaptive autodiff stack's size should have been determined.");

    const auto &var_name = stmt->raw_name();
    emit("Ti_u8 {}[{}];", var_name, stmt->size_in_bytes() + sizeof(uint32_t));
    emit("Ti_ad_stack_init({});", var_name);
  }

  void visit(AdStackPopStmt *stmt) override {
    emit("Ti_ad_stack_pop({});", stmt->stack->raw_name());
  }

  void visit(AdStackPushStmt *stmt) override {
    auto *stack = stmt->stack->as<AdStackAllocaStmt>();
    const auto &stack_name = stack->raw_name();
    auto elem_size = stack->element_size_in_bytes();
    emit("Ti_ad_stack_push({}, {});", stack_name, elem_size);
    auto primal_name = stmt->raw_name() + "_primal_";
    auto dt_name = cc_data_type_name(stmt->element_type());
    auto var = define_var(dt_name + " *", primal_name);
    emit("{} = ({} *) Ti_ad_stack_top_primal({}, {});", var, dt_name,
         stack_name, elem_size);
    emit("*{} = {};", primal_name, stmt->v->raw_name());
  }

  void visit(AdStackLoadTopStmt *stmt) override {
    auto *stack = stmt->stack->as<AdStackAllocaStmt>();
    const auto primal_name = stmt->raw_name() + "_primal_";
    auto dt_name = cc_data_type_name(stmt->element_type());
    auto var = define_var(dt_name + " *", primal_name);
    emit("{} = ({} *)Ti_ad_stack_top_primal({}, {});", var, dt_name,
         stack->raw_name(), stack->element_size_in_bytes());
    emit("{} = *{};", define_var(dt_name, stmt->raw_name()), primal_name);
  }

  void visit(AdStackLoadTopAdjStmt *stmt) override {
    auto *stack = stmt->stack->as<AdStackAllocaStmt>();
    const auto adjoint_name = stmt->raw_name() + "_adjoint_";
    auto dt_name = cc_data_type_name(stmt->element_type());
    auto var = define_var(dt_name + " *", adjoint_name);
    emit("{} = ({} *)Ti_ad_stack_top_adjoint({}, {});", var, dt_name,
         stack->raw_name(), stack->element_size_in_bytes());
    emit("{} = *{};", define_var(dt_name, stmt->raw_name()), adjoint_name);
  }

  void visit(AdStackAccAdjointStmt *stmt) override {
    auto *stack = stmt->stack->as<AdStackAllocaStmt>();
    const auto adjoint_name = stmt->raw_name() + "_adjoint_";
    auto dt_name = cc_data_type_name(stmt->element_type());
    auto var = define_var(dt_name + " *", adjoint_name);
    emit("{} = ({} *)Ti_ad_stack_top_adjoint({}, {});", var, dt_name,
         stack->raw_name(), stack->element_size_in_bytes());
    emit("*{} += {};", adjoint_name, stmt->v->raw_name());
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    line_appender.append(std::move(f), std::move(args)...);
  }

  template <typename... Args>
  void emit_header(std::string f, Args &&... args) {
    line_appender_header.append(std::move(f), std::move(args)...);
  }
};  // namespace cccp

std::unique_ptr<CCKernel> CCKernelGen::compile() {
  auto layout = cc_program_impl_->get_layout();
  CCTransformer tran(kernel_, layout);

  tran.run();
  auto source = tran.get_source();
  auto ker = std::make_unique<CCKernel>(cc_program_impl_, kernel_, source,
                                        kernel_->name);
  ker->compile();
  return ker;
}

}  // namespace cccp
TLANG_NAMESPACE_END
