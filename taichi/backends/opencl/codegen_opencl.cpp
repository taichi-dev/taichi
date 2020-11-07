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
    return std::make_unique<OpenclKernel>(program, kernel,
        offload_count, source);
  }

 private:
  LineAppender line_appender;
  bool is_top_level{true};
  GetRootStmt *root_stmt{nullptr};

  int offload_count{0};

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
    TI_WARN("Unsupported statement `{}` for OpenCL", typeid(*stmt).name());
  }

  void visit(OffloadedStmt *stmt) override {
    auto kernel_name = fmt::format("{}_k{}", kernel->name, offload_count);
    emit("");
    emit("__kernel void {}(", kernel_name);
    emit("    __global struct Ti_S0root *root");
    emit("    ) {{");

    TI_ASSERT(is_top_level);
    is_top_level = false;
    if (stmt->task_type == OffloadedStmt::TaskType::serial) {
      generate_serial_kernel(stmt);
    } else if (stmt->task_type == OffloadedStmt::TaskType::range_for) {
      generate_range_for_kernel(stmt);
    } else {
      TI_ERROR("Unsupported offload type={} on OpenCL backend",
               stmt->task_name());
    }
    is_top_level = true;

    emit("}}");
    emit("");
    offload_count++;
  }

  void generate_serial_kernel(OffloadedStmt *stmt) {
    emit("  /* serial kernel */");
    stmt->body->accept(this);
  }

  void generate_range_for_kernel(OffloadedStmt *stmt) {
    emit("  /* range-for kernel */");
    ScopedIndent _s(line_appender);

    auto name = stmt->raw_name();

    TI_ASSERT(stmt->const_begin && stmt->const_end);
    emit("Ti_i32 {}_beg = {};", name, stmt->begin_value);
    emit("Ti_i32 {}_end = {};", name, stmt->end_value);

    emit("for (Ti_i32 {} = {}_beg + (Ti_i32)get_global_id(0);", name, name);
    emit("    {} < {}_end; {} += (Ti_i32)get_global_size(0)) {{", name, name, name);
    //emit(R"(printf("What?\n%llu %llu\n", get_global_id(0), get_global_size(0));)");
    stmt->body->accept(this);
    emit("}}");
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

  void run() {
    emit("{}", program->get_header_lines());
    emit("/* Generated OpenCL program of Taichi kernel: {} */", kernel->name);
    kernel->ir->accept(this);
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
