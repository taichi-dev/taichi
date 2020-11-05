#include "opencl_program.h"
#include "opencl_kernel.h"

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/util/line_appender.h"
#include "taichi/util/macros.h"

TLANG_NAMESPACE_BEGIN
namespace opencl {

namespace {

// Generate corresponding OpenCL Source Code for Taichi Kernels
class OpenclKernelGen : public IRVisitor {
 private:
  Kernel *kernel;

 public:
  OpenclKernelGen(Kernel *kernel) : kernel(kernel) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  std::unique_ptr<OpenclKernel> compile() {
    this->lower();
    this->run();
    return std::make_unique<OpenclKernel>("func", "hello");
  }

 private:
  LineAppender line_appender;
  bool is_top_level{true};

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
    TI_WARN("[cl] unsupported statement type {}", typeid(*stmt).name());
  }

  void run() {
    emit("void Tk_{}() {{", kernel->name);
    kernel->ir->accept(this);
    emit("}}");
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

OpenclProgram::OpenclProgram(Program *prog) : prog(prog) {
}

OpenclProgram::~OpenclProgram() {
}

FunctionType OpenclProgram::compile_kernel(Kernel *kernel) {
  OpenclKernelGen codegen(kernel);
  auto ker = codegen.compile();
  auto ker_ptr = ker.get();
  kernels.push_back(std::move(ker));  // prevent unique_ptr being released
  return [ker_ptr](Context &ctx) { return ker_ptr->launch(&ctx); };
}

}  // namespace opencl
TLANG_NAMESPACE_END
