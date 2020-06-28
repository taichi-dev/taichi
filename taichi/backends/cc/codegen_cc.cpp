#include "codegen_cc.h"
#include "cc_kernel.h"
#include "cc_layout.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/util/line_appender.h"
#include "cc_utils.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {  // Codegen for C Compiler Processor

class CCTransformer : public IRVisitor {
private:
  [[maybe_unused]] Kernel *kernel;
  [[maybe_unused]] CCLayout *layout;

  LineAppender line_appender;
  LineAppender line_appender_header;
  bool is_top_level{true};

public:
  CCTransformer(Kernel *kernel, CCLayout *layout)
      : kernel(kernel), layout(layout)
  {
      allow_undefined_visitor = true;
      invoke_default_visitor = true;
  }

  void run() {
    this->lower_ast();
    emit_header("#include <stdio.h>");
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

  void visit(ConstStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("{} {} = {};", cc_data_type_name(stmt->element_type()),
         stmt->raw_name(), stmt->val[0].stringify());
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
    emit_header("void {}(void);", get_sym_name(kernel->name));
    emit("void {}(void) {{", get_sym_name(kernel->name));
    {
      ScopedIndent _s(line_appender);
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
  auto layout = kernel->program.cc_program->layout.get();
  CCTransformer tran(kernel, layout);

  tran.run();
  auto source = tran.get_source();
  auto ker = std::make_unique<CCKernel>(source, kernel->name);
  ker->compile();
  return ker;
}

FunctionType compile_kernel(Kernel *kernel) {
  CCKernelGen codegen(kernel);
  auto compiled = codegen.compile();
  auto compiled_ptr = compiled.get();
  auto program = kernel->program.cc_program.get();
  program->kernels.push_back(std::move(compiled));
  return [program, compiled_ptr] (Context &ctx) {
    return program->launch(compiled_ptr, &ctx);
  };
}

}  // namespace cccp
TLANG_NAMESPACE_END
