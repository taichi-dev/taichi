#include "codegen_cc.h"
#include "cc_kernel.h"
#include "cc_layout.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/util/line_appender.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {  // Codegen for C Compiler Processor

class CCTransformer : public IRVisitor {
private:
  Kernel *kernel;
  CCLayout *layout;

  LineAppender line_appender;
  LineAppender line_appender_header;
  bool is_top_level{false};

public:
  CCTransformer(Kernel *kernel, CCLayout *layout)
      : kernel(kernel), layout(layout)
  {
      allow_undefined_visitor = true;
      invoke_default_visitor = true;
  }

  void run() {
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

  void generate_serial_kernel(OffloadedStmt *stmt) {
    emit("int main(void) {{");
    {
      ScopedIndent _s(line_appender);
      emit("printf(\"Hello, world!\\n\");");
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
};

std::unique_ptr<CCKernel> CCKernelGen::compile() {
  CCTransformer tran(kernel, layout);

  auto source = "#include <stdio.h>\n"
    "int main(void) {\n"
    "\tprintf(\"Hello, world!\\n\");\n"
    "}\n";
  auto ker = std::make_unique<CCKernel>(source);
  return ker;
}

FunctionType compile_kernel(
      Program *program,
      Kernel *kernel,
      CCLayout *layout,
      CCLauncher *launcher) {
  CCKernelGen codegen(program, kernel, layout);
  auto compiled = codegen.compile();
  auto compiled_ptr = compiled.get();
  launcher->keep(std::move(compiled));
  return [launcher, compiled_ptr] (Context &ctx) {
      return launcher->launch(compiled_ptr, &ctx);
    };
}

}  // namespace cccp
TLANG_NAMESPACE_END
