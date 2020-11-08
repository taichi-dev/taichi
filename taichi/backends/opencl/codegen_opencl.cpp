#include "opencl_program.h"
#include "opencl_kernel.h"
#include "opencl_utils.h"

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/util/line_appender.h"
#include "taichi/util/macros.h"
#include "taichi/util/str.h"

TLANG_NAMESPACE_BEGIN
namespace opencl {

namespace {

std::string opencl_get_node_type_name(SNode *snode) {
  return fmt::format("struct Ti_{}", snode->get_node_type_name_hinted());
}

// Generate corresponding OpenCL Source Code for Taichi Kernels
class OpenclKernelGen : public IRVisitor {
 private:
  OpenclProgram *program;
  Kernel *kernel;

 public:
  OpenclKernelGen(OpenclProgram *program, Kernel *kernel)
      : program(program), kernel(kernel) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  std::unique_ptr<OpenclKernel> compile() {
    this->lower();
    this->run();
    auto source = line_appender.lines();
    TI_INFO("[{}]:\n{}", kernel->name, source);
    return std::make_unique<OpenclKernel>(program, kernel, offloads, source);
  }

 private:
  LineAppender line_appender;
  bool is_top_level{true};
  GetRootStmt *root_stmt{nullptr};

  std::vector<OpenclOffloadMeta> offloads;

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    line_appender.append(std::move(f), std::move(args)...);
  }

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
    TI_ERROR("Unsupported statement `{}` for OpenCL", typeid(*stmt).name());
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt_name = opencl_data_type_name(stmt->element_type().ptr_removed());
    emit("__global {} *{} = (__global {} *)(gtmp + {});", dt_name,
        stmt->raw_name(), dt_name, stmt->offset);
  }

  void visit(KernelReturnStmt *stmt) override {
    emit("*(__global {} *)gtmp = {};",
         opencl_data_type_name(stmt->element_type()), stmt->value->raw_name());
  }

  void visit(ArgLoadStmt *stmt) override {
    if (stmt->is_ptr) {
      auto dt_name = opencl_data_type_name(stmt->element_type().ptr_removed());
      emit("__global {} *{} = arg{};", dt_name, stmt->raw_name(), stmt->arg_id);
    } else {
      auto dt_name = opencl_data_type_name(stmt->element_type());
      emit("{} {} = arg{};", dt_name, stmt->raw_name(), stmt->arg_id);
    }
  }

  void generate_kernel_arguments() {
    for (int i = 0; i < kernel->args.size(); i++) {
      auto dt_name = opencl_data_type_name(kernel->args[i].dt);
      if (kernel->args[i].is_nparray) {
        emit("    , __global {} *arg{}", dt_name, i);
      } else {
        emit("    , {} arg{}", dt_name, i);
      }
    }
    for (int i = 0; i < kernel->args.size(); i++) {
      if (kernel->args[i].is_nparray) {
        static_assert(taichi_max_num_indices == 8);
        emit("    , int8 arg{}_shape", i);
      }
    }
  }

  OpenclOffloadMeta make_offload_meta(
      OffloadedStmt *stmt, std::string kernel_name, size_t global_dim) {
    OpenclOffloadMeta meta;

    meta.kernel_name = kernel_name;
    meta.block_dim = stmt->block_dim;

    if (stmt->grid_dim != 0) {
      TI_ASSERT(meta.block_dim != 0);
      meta.global_dim = stmt->grid_dim * meta.block_dim;

    } else {
      if (meta.block_dim != 0) {
        // OpenCL seems require |global_dim| to be a *integral multiple* of
        // |block_dim| when |block_dim| is explicitly specified (non-zero):
        global_dim = (global_dim + meta.block_dim - 1)
          / meta.block_dim * meta.block_dim;
      }
      meta.global_dim = global_dim;
    }
    return meta;
  }

  void visit(OffloadedStmt *stmt) override {
    auto kernel_name = fmt::format("{}_k{}", kernel->name, offloads.size());

    emit("");
    emit("__kernel void {}", kernel_name);
    emit("    ( __global struct Ti_S0root *root");
    emit("    , __global uchar *gtmp");
    generate_kernel_arguments();
    emit("    ) {{");
    emit("  ulong rand_seed = ++*(__global int *)(gtmp + 4096) + get_global_id(0);");

    TI_ASSERT(is_top_level);
    is_top_level = false;

    size_t global_dim = 1;
    if (stmt->task_type == OffloadedStmt::TaskType::serial) {
      global_dim = generate_serial_kernel(stmt);
    } else if (stmt->task_type == OffloadedStmt::TaskType::range_for) {
      global_dim = generate_range_for_kernel(stmt);
    } else {
      TI_ERROR("Unsupported offload type={} on OpenCL backend",
               stmt->task_name());
    }

    is_top_level = true;

    emit("}}");
    emit("");

    offloads.push_back(make_offload_meta(stmt, kernel_name, global_dim));
  }

  size_t generate_serial_kernel(OffloadedStmt *stmt) {
    emit("  /* serial kernel */");
    stmt->body->accept(this);
    return 1;
  }

  size_t generate_range_for_kernel(OffloadedStmt *stmt) {
    emit("  /* range-for kernel */");

    ScopedIndent _s(line_appender);
    auto name = stmt->raw_name();

    if (stmt->const_begin)
      emit("int {}_beg = {};", name, stmt->begin_value);
    else
      emit("int {}_beg = *(__global int *)(gtmp + {});", name, stmt->begin_offset);

    if (stmt->const_end)
      emit("int {}_end = {};", name, stmt->end_value);
    else
      emit("int {}_end = *(__global int *)(gtmp + {});", name, stmt->end_offset);

    emit("for (int {} = {}_beg + get_global_id(0);", name, name);
    emit("    {} < {}_end; {} += get_global_size(0)) {{", name, name, name);
    stmt->body->accept(this);
    emit("}}");

    if (stmt->const_begin && stmt->const_end) {
      if (stmt->end_value > stmt->begin_value)
        return stmt->end_value - stmt->begin_value;
      else
        return 1;
    } else {
      return 4096;  // default global_dim for dynamic range-for
    }
  }

  void visit(RangeForStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    if (!stmt->reversed) {
      emit("for (int {} = {}; {} < {}; {}++) {{", stmt->raw_name(),
          stmt->begin->raw_name(), stmt->raw_name(), stmt->end->raw_name(),
          stmt->raw_name());
    } else {
      // reversed for loop
      emit("for (int {} = {} - 1; {} >= {}; {}--) {{", stmt->raw_name(),
          stmt->end->raw_name(), stmt->raw_name(), stmt->begin->raw_name(),
          stmt->raw_name());
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

  void make_random_uint(std::string name) {
    // Java random: https://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl
    emit("rand_seed = (rand_seed * 0x5deece66dl + 0xbl) & ((1l << 48) - 1);");
    // Wang hash: https://software.intel.com/content/www/us/en/develop/articles/parallel-noise-and-random-functions-for-opencl-kernels.html
    emit("uint {} = rand_seed >> 16;", name);
    //emit("{} = {} * 1103515245 + 12345;", name, name);
    emit("{} = 9 * (({} ^ 61) ^ ({} >> 16));", name, name, name);
    emit("{} = 0x27d4eb2d * ({} ^ ({} << 4));", name, name, name);
	  emit("{} ^= {} >> 15;", name, name);
  }

  void visit(RandStmt *stmt) override {
    auto name = stmt->raw_name();
    auto dt_name = opencl_data_type_name(stmt->ret_type);

    if (stmt->ret_type->is_primitive(PrimitiveTypeID::i64)) {
      make_random_uint(fmt::format("{}_low", name));
      make_random_uint(fmt::format("{}_high", name));
      emit("{} {} = ((ulong){}_high << 32) | {}_low;",
          dt_name, name, name, name);

    } else if (is_integral(stmt->ret_type)) {
      make_random_uint(fmt::format("{}_uint", name));
      emit("{} {} = ({}){}_uint;", dt_name, name, dt_name, name);

    } else {
      make_random_uint(fmt::format("{}_uint", name));
      emit("{} {} = ({}){}_uint * ({})(1.0 / 4294967296.0);",
          dt_name, name, dt_name, name, dt_name);
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

  void visit(LoopIndexStmt *stmt) override {
    TI_ASSERT(stmt->index == 0);
    if (stmt->loop->is<OffloadedStmt>()) {
      auto type = stmt->loop->as<OffloadedStmt>()->task_type;
      if (type == OffloadedStmt::TaskType::range_for) {
        emit("int {} = {};", stmt->raw_name(), stmt->loop->raw_name());
      } else {
        TI_NOT_IMPLEMENTED
      }

    } else if (stmt->loop->is<RangeForStmt>()) {
      emit("int {} = {};", stmt->raw_name(), stmt->loop->raw_name());

    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(ConstStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("{} {} = {};",
         opencl_data_type_name(stmt->element_type()), stmt->raw_name(),
         stmt->val[0].stringify());
  }

  void visit(AllocaStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("{} {} = 0;",
         opencl_data_type_name(stmt->element_type()), stmt->raw_name());
  }

  void visit(LocalLoadStmt *stmt) override {
    bool linear_index = true;
    for (int i = 0; i < (int)stmt->ptr.size(); i++) {
      if (stmt->ptr[i].offset != i) {
        linear_index = false;
      }
    }
    TI_ASSERT(stmt->same_source() && linear_index &&
              stmt->width() == stmt->ptr[0].var->width());

    emit("{} {} = {};", opencl_data_type_name(stmt->element_type()),
        stmt->raw_name(), stmt->ptr[0].var->raw_name());
  }

  void visit(LocalStoreStmt *stmt) override {
    emit("{} = {};", stmt->ptr->raw_name(), stmt->data->raw_name());
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
    // https://www.informit.com/articles/article.aspx?p=1732873&seqNum=3
    emit("{} {} = arg{}_shape.s{};", opencl_data_type_name(stmt->element_type()),
        stmt->raw_name(), stmt->arg_id, stmt->axis);
  }

  void visit(ExternalPtrStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    std::string offset = "0";
    const auto *argload = stmt->base_ptrs[0]->as<ArgLoadStmt>();
    int arg_id = argload->arg_id;
    for (int i = 0; i < stmt->indices.size(); i++) {
      auto stride = fmt::format("arg{}_shape.s{}", arg_id, i);
      offset = fmt::format("({} * {} + {})", offset, stride,
                           stmt->indices[i]->raw_name());
    }
    auto dt_name = opencl_data_type_name(stmt->element_type().ptr_removed());
    emit("__global {} *{} = {} + {};", dt_name, stmt->raw_name(),
        stmt->base_ptrs[0]->raw_name(), offset);
  }

  void visit(BitExtractStmt *stmt) override {
    emit("int {} = (({} >> {}) & ((1 << {}) - 1));",
         stmt->raw_name(), stmt->input->raw_name(),
         stmt->bit_begin, stmt->bit_end - stmt->bit_begin);
  }

  void visit(GetRootStmt *stmt) override {
    auto root = kernel->program.snode_root.get();
    emit("__global {} *{} = root;",  // |root| is passed as a kernel argument
         opencl_get_node_type_name(root), stmt->raw_name());
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

    emit("__global {} *{} = &{}[{}];",
         opencl_get_node_type_name(stmt->snode), stmt->raw_name(),
         input_ptr->raw_name(), stmt->input_index->raw_name());
  }

  void visit(GetChStmt *stmt) override {
    auto snode = stmt->output_snode;
    std::string type;
    if (snode->type == SNodeType::place) {
      emit("__global {} *{} = &{}->{};",
          opencl_data_type_name(snode->dt), stmt->raw_name(),
          stmt->input_ptr->raw_name(), snode->get_node_type_name());
    } else {
      emit("__global {} *{} = {}->{};",
          opencl_get_node_type_name(snode), stmt->raw_name(),
          stmt->input_ptr->raw_name(), snode->get_node_type_name());
    }
  }

  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("{} {} = *{};",
         opencl_data_type_name(stmt->element_type()), stmt->raw_name(),
         stmt->ptr->raw_name());
  }

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("*{} = {};", stmt->ptr->raw_name(), stmt->data->raw_name());
  }

  void visit(UnaryOpStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt_name = opencl_data_type_name(stmt->element_type());
    auto operand_name = stmt->operand->raw_name();
    auto dest_name = stmt->raw_name();
    auto type = stmt->element_type();
    auto op = unary_op_type_symbol(stmt->op_type);

    if (is_real(type) && stmt->op_type == UnaryOpType::abs)
      op = "fabs";

    if (stmt->op_type == UnaryOpType::cast_value) {
      emit("{} {} = ({}) {};", dt_name, dest_name, dt_name, operand_name);

    } else if (stmt->op_type == UnaryOpType::cast_bits) {
      auto operand_dt_name =
          opencl_data_type_name(stmt->operand->element_type());
      emit("union {{ {} bc_src; {} bc_dst; }} {}_bitcast;", operand_dt_name,
           dt_name, dest_name);
      emit("{}_bitcast.bc_src = {};", dest_name, operand_name);
      emit("{} {} = {}_bitcast.bc_dst;", dt_name, dest_name, dest_name);

    } else if (opencl_is_unary_op_infix(stmt->op_type)) {
      emit("{} {} = {}{};", dt_name, dest_name, op, operand_name);

    } else if (stmt->op_type == UnaryOpType::sgn) {
      emit("{} {} = {} > 0 ? 1 : {} < 0 ? -1 : 0;",
          dt_name, dest_name, operand_name, operand_name);

    } else {
      emit("{} {} = {}({});", dt_name, dest_name, op, operand_name);
    }
  }

  void visit(BinaryOpStmt *bin) override {
    TI_ASSERT(bin->width() == 1);
    auto dt_name = opencl_data_type_name(bin->element_type());
    auto lhs_name = bin->lhs->raw_name();
    auto rhs_name = bin->rhs->raw_name();
    auto bin_name = bin->raw_name();
    auto type = bin->element_type();
    auto binop = binary_op_type_symbol(bin->op_type);

    // floordiv should have been demoted in demote_operations.cpp
    TI_ASSERT(bin->op_type != BinaryOpType::floordiv);

    if (is_real(type) && (bin->op_type == BinaryOpType::min
          || bin->op_type == BinaryOpType::max))
      binop = "f" + binop;

    if (opencl_is_binary_op_infix(bin->op_type)) {

      if (is_comparison(bin->op_type)) {
        // XXX(#577): Taichi uses -1 as true due to LLVM i1...
        emit("{} {} = -({} {} {});", dt_name, bin_name,
            lhs_name, binop, rhs_name);

      } else if (bin->op_type == BinaryOpType::mod) {
        // NOTE: the OpenCL built-in function `fmod()` is a pythonic mod:
        // https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/fmod.html
        // but |BinaryOpType::mod| means a C-style mod in Taichi...
        emit("{} {} = {} - {} * (int)({} / {});", dt_name, bin_name, lhs_name,
             rhs_name, lhs_name, rhs_name);

      } else if (bin->op_type == BinaryOpType::truediv) {
        emit("{} {} = ({}) {} / {};", dt_name, bin_name,
            dt_name, lhs_name, rhs_name);

      } else {
        emit("{} {} = {} {} {};", dt_name, bin_name, lhs_name, binop, rhs_name);
      }

    } else if (bin->op_type == BinaryOpType::pow) {
      if (is_integral(bin->rhs->element_type())) {
        if (is_integral(bin->lhs->element_type())) {
          auto ret = bin_name;
          auto x = bin_name + "_x";
          auto y = bin_name + "_y";
          emit("{} {} = 0;", dt_name, ret);
          emit("{} {} = {};", dt_name, x, lhs_name);
          emit("{} {} = {};", dt_name, y, rhs_name);
          emit("if ({} >= 0) {{", y);
          emit("  {} = 1;", ret);
          emit("  while ({}) {{", y);
          emit("    if ({} & 1)", y);
          emit("      {} *= {};", ret, x);
          emit("    {} *= {};", x, x);
          emit("    {} >>= 1;", y);
          emit("  }}");
          emit("}}");

        } else {
          emit("{} {} = ({}) pown({}, {});", dt_name, bin_name, dt_name,
              lhs_name, rhs_name);
        }
      } else {
        emit("{} {} = ({}) pow(({}) {}, ({}) {});", dt_name,
            bin_name, dt_name, dt_name, lhs_name, dt_name, rhs_name);
      }

    } else {
      emit("{} {} = {}({}, {});", dt_name, bin_name, binop, lhs_name, rhs_name);
    }
  }

  void visit(TernaryOpStmt *tri) override {
    TI_ASSERT(tri->op_type == TernaryOpType::select);
    emit("{} {} = {} ? {} : {};", opencl_data_type_name(tri->element_type()),
         tri->raw_name(), tri->op1->raw_name(), tri->op2->raw_name(),
         tri->op3->raw_name());
  }

  void visit(AtomicOpStmt *stmt) override {
    auto name = stmt->raw_name();
    auto dst_name = stmt->dest->raw_name();
    auto src_name = stmt->val->raw_name();
    auto op_name = opencl_atomic_op_type_name(stmt->op_type);
    auto type = stmt->dest->element_type().ptr_removed();
    auto dt_name = opencl_data_type_name(type);

    if (is_integral(type)) {
      emit("{} {} = atomic_{}({}, {});", dt_name, name, op_name,
          dst_name, src_name);

    } else {
      // https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/#:~:text=In%20OpenCL%20there%20is%20only%20atomic_add%20or%20atomic_mul,%28%29.%20Here%20is%20his%20example%20for%20atomic%20add%3A
      auto int_dt_name = opencl_data_type_name(real_to_integral(type));
      emit("union {{");
      emit("  {} f;", dt_name);
      emit("  {} i;", int_dt_name);
      emit("}} {}_old, {}_new;", name, name);
      emit("do {{");
      emit("  {}_old.f = *{};", name, dst_name);
      auto bin_op = atomic_to_binary_op_type(stmt->op_type);
      auto op_name = binary_op_type_symbol(bin_op);
      if (opencl_is_binary_op_infix(bin_op)) {
        emit("  {}_new.f = {}_old.f {} {};", name, name, op_name, src_name);
      } else {
        emit("  {}_new.f = {}({}_old.f, {});", name, op_name, name, src_name);
      }
      emit("}} while ({}_old.i != atomic_cmpxchg(", name);
      emit("    (volatile __global {} *){},", int_dt_name, dst_name);
      emit("    {}_old.i, {}_new.i));", name, name);
      emit("{} {} = {}_old.f;", dt_name, name, name);
    }
  }

  void run() {
    TI_TRACE("start running OpenCL codegen for {}", kernel->name);
    emit("{}", program->get_header_lines());
    emit("/* Generated OpenCL program for Taichi kernel: {} */", kernel->name);
    kernel->ir->accept(this);
    TI_TRACE("done running OpenCL codegen for {}", kernel->name);
  }

  void lower() {
    auto ir = kernel->ir.get();
    auto config = kernel->program.config;
    config.demote_dense_struct_fors = true;
    irpass::compile_to_executable(ir, config,
                                  /*vectorize=*/false, kernel->grad,
                                  /*ad_use_stack=*/false, config.print_ir,
                                  /*lower_global_access*/ true);
  }
};

}  // namespace

bool OpenclProgram::is_opencl_api_available() {
  return true;
}

std::string OpenclProgram::get_header_lines() {
  std::string header_source =
#include "taichi/backends/opencl/runtime/base.h"
    ;
  return header_source + "\n" + layout_source;
}

FunctionType OpenclProgram::compile_kernel(Kernel *kernel) {
  OpenclKernelGen codegen(this, kernel);
  auto ker = codegen.compile();
  auto ker_ptr = ker.get();
  kernels.push_back(std::move(ker));  // prevent unique_ptr being released
  return [ker_ptr](Context &ctx) { return ker_ptr->launch(&ctx); };
}

}  // namespace opencl
TLANG_NAMESPACE_END
