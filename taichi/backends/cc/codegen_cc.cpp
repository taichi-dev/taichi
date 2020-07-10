#include "codegen_cc.h"
#include "cc_kernel.h"
#include "cc_layout.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/util/line_appender.h"
#include "taichi/util/str.h"
#include "cc_utils.h"

#define C90_COMPAT 0

TLANG_NAMESPACE_BEGIN
namespace cccp {  // Codegen for C Compiler Processor

namespace {
std::string get_node_ptr_name(SNode *snode) {
  return fmt::format("struct {} *", snode->get_node_type_name_hinted());
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
    kernel->ir->accept(this);
  }

  void lower_ast() {
    auto ir = kernel->ir.get();
    auto config = kernel->program.config;
    config.demote_dense_struct_fors = true;
    irpass::compile_to_offloads(ir, config,
                                /*vectorize=*/false, kernel->grad,
                                /*ad_use_stack=*/false, config.print_ir,
                                /*lower_global_access*/ true);
  }

  std::string get_source() {
    return line_appender_header.lines() + "\n" + line_appender.lines();
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
         define_var("int", stmt->raw_name()), stmt->input->raw_name(),
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
    auto root = kernel->program.snode_root.get();
    emit("{} = RTi_get_root();",
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
         stmt->ptr->raw_name());
  }

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("*{} = {};", stmt->ptr->raw_name(), stmt->data->raw_name());
  }

  void visit(ConstStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("{} = {};",
         define_var(cc_data_type_name(stmt->element_type()), stmt->raw_name()),
         stmt->val[0].stringify());
  }

  static std::string _get_libc_function_name(std::string name, DataType dt) {
    switch (dt) {
      case DataType::i32:
        return name;
      case DataType::i64:
        return "ll" + name;
      case DataType::f32:
        return name + "f";
      case DataType::f64:
        return name;
      default:
        TI_ERROR("Unsupported function \"{}\" for DataType={} on C backend",
                 name, data_type_name(dt));
    }
  }

  static std::string get_libc_function_name(std::string name, DataType dt) {
    name = _get_libc_function_name(name, dt);
    if (name == "max" || name == "min" || name == "abs") {
      if (is_real(dt)) {
        name = "f" + name;
      } else if (name != "abs") {
        name = "RTi_" + name;
      }
    }
    return name;
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
      emit("{} = {} {} {};", var, lhs_name, binop, rhs_name);
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
    const auto type = stmt->element_type();
    auto var = define_var(cc_data_type_name(type), stmt->raw_name());
    emit("{} = *{};", var, dest_ptr);
    if (stmt->op_type == AtomicOpType::max ||
        stmt->op_type == AtomicOpType::min) {
      emit("*{} = {};", invoke_libc(op, type, "*{}, {}", dest_ptr, src_name));
    } else {
      emit("*{} {}= {};", dest_ptr, op, src_name);
    }
  }

  void visit(LinearizeStmt *stmt) override {
    std::string val = "0";
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      val = fmt::format("({} * {} + {})", val, stmt->strides[i],
                        stmt->inputs[i]->raw_name());
    }
    emit("{} = {};", define_var("int", stmt->raw_name()), val);
  }

  void visit(PrintStmt *stmt) override {
    std::string format;
    std::vector<std::string> values;

    for (int i = 0; i < stmt->contents.size(); i++) {
      auto const &content = stmt->contents[i];

      if (std::holds_alternative<Stmt *>(content)) {
        auto arg_stmt = std::get<Stmt *>(content);
        format += data_type_format(arg_stmt->ret_type.data_type);
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
    emit_header("void Ti_{}(void) {{", kernel->name);
    {
      ScopedIndent _s1(line_appender);
      ScopedIndent _s2(line_appender_header);
      stmt->body->accept(this);
    }
    emit("}}");
  }

  void visit(OffloadedStmt *stmt) override {
    TI_ASSERT(is_top_level);
    is_top_level = false;
    if (stmt->task_type == OffloadedStmt::TaskType::serial) {
      generate_serial_kernel(stmt);
    } else {
      TI_ERROR("[glsl] Unsupported offload type={} on C backend",
               stmt->task_name());
    }
    is_top_level = true;
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    line_appender.append(std::move(f), std::move(args)...);
  }

  template <typename... Args>
  void emit_header(std::string f, Args &&... args) {
    line_appender_header.append(std::move(f), std::move(args)...);
  }
};

std::unique_ptr<CCKernel> CCKernelGen::compile() {
  auto program = kernel->program.cc_program.get();
  auto layout = program->layout.get();
  CCTransformer tran(kernel, layout);

  tran.run();
  auto source = tran.get_source();
  auto ker = std::make_unique<CCKernel>(program, source, kernel->name);
  ker->compile();
  return ker;
}

FunctionType compile_kernel(Kernel *kernel) {
  CCKernelGen codegen(kernel);
  auto ker = codegen.compile();
  auto ker_ptr = ker.get();
  auto program = kernel->program.cc_program.get();
  program->add_kernel(std::move(ker));
  return [ker_ptr](Context &ctx) { return ker_ptr->launch(&ctx); };
}

}  // namespace cccp
TLANG_NAMESPACE_END
