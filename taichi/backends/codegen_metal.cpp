#include "codegen_metal.h"

#include <taichi/ir.h>

#ifdef TC_SUPPORTS_METAL

TLANG_NAMESPACE_BEGIN
namespace metal {
namespace {

class MetalKernelCodegen : public IRVisitor {
 public:
  MetalKernelCodegen(const std::string &mtl_kernel_prefix)
      : mtl_kernel_prefix_(mtl_kernel_prefix) {
    allow_undefined_visitor = true;
  }

  const std::string &kernel_source_code() const {
    return kernel_src_code_;
  }

  const std::vector<MetalKernelAttributes> &kernels_attribs() const {
    return mtl_kernels_attribs_;
  }

  void run(const std::string &snode_structs_source_code, Kernel *kernel) {
    generate_mtl_header(snode_structs_source_code);
    generate_kernel_args_struct(kernel);
    kernel->ir->accept(this);
  }

 private:
  void generate_mtl_header(const std::string &snode_structs_source_code) {
    emit("#include <metal_stdlib>");
    emit("using namespace metal;");
    emit("");
    emit("namespace {{");
    kernel_src_code_ += snode_structs_source_code;
    emit("}}  // namespace");
    emit("");
  }

  void generate_kernel_args_struct(Kernel *kernel) {
    TC_NOT_IMPLEMENTED
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    kernel_src_code_ +=
        indent_ + fmt::format(f, std::forward<Args>(args)...) + "\n";
  }

  const std::string mtl_kernel_prefix_;

  bool is_top_level_{true};
  int mtl_kernel_count_{0};
  std::vector<MetalKernelAttributes> mtl_kernels_attribs_;
  MetalKernelArgsAttributes args_attribs_;
  GetRootStmt *root_stmt_{nullptr};
  MetalKernelAttributes *current_kernel_attribs_{nullptr};
  std::string kernel_src_code_;
  std::string indent_;
};

}  // namespace

MetalCodeGen::MetalCodeGen(const std::string &kernel_name,
                           const StructCompiledResult *struct_compiled)
    : id_(CodeGenBase::get_kernel_id()),
      taichi_kernel_name_(fmt::format("mtl_k{:04d}_{}", id_, kernel_name)),
      struct_compiled_(struct_compiled) {
}

FunctionType MetalCodeGen::compile(Program &,
                                   Kernel &kernel,
                                   MetalRuntime *runtime) {
  this->prog_ = &kernel.program;
  this->kernel_ = &kernel;
  lower();
  return gen(runtime);
}

void MetalCodeGen::lower() {
  auto ir = kernel_->ir;
  const bool print_ir = prog_->config.print_ir;
  if (print_ir) {
    TC_TRACE("Initial IR:");
    irpass::print(ir);
  }

  if (kernel_->grad) {
    irpass::reverse_segments(ir);
    irpass::re_id(ir);
    if (print_ir) {
      TC_TRACE("Segment reversed (for autodiff):");
      irpass::print(ir);
    }
  }

  irpass::lower(ir);
  irpass::re_id(ir);
  if (print_ir) {
    TC_TRACE("Lowered:");
    irpass::print(ir);
  }

  irpass::typecheck(ir);
  irpass::re_id(ir);
  if (print_ir) {
    TC_TRACE("Typechecked:");
    irpass::print(ir);
  }

  irpass::demote_dense_struct_fors(ir);
  irpass::typecheck(ir);
  if (print_ir) {
    TC_TRACE("Dense Struct-for demoted:");
    irpass::print(ir);
  }

  irpass::constant_fold(ir);
  if (prog_->config.simplify_before_lower_access) {
    irpass::simplify(ir);
    irpass::re_id(ir);
    if (print_ir) {
      TC_TRACE("Simplified I:");
      irpass::print(ir);
    }
  }

  if (kernel_->grad) {
    irpass::demote_atomics(ir);
    irpass::full_simplify(ir);
    irpass::typecheck(ir);
    if (print_ir) {
      TC_TRACE("Before make_adjoint:");
      irpass::print(ir);
    }
    irpass::make_adjoint(ir);
    if (print_ir) {
      TC_TRACE("After make_adjoint:");
      irpass::print(ir);
    }
    irpass::typecheck(ir);
  }

  irpass::lower_access(ir, prog_->config.use_llvm);
  irpass::re_id(ir);
  if (print_ir) {
    TC_TRACE("Access Lowered:");
    irpass::print(ir);
  }

  irpass::die(ir);
  irpass::re_id(ir);
  if (print_ir) {
    TC_TRACE("DIEd:");
    irpass::print(ir);
  }

  irpass::flag_access(ir);
  irpass::re_id(ir);
  if (print_ir) {
    TC_TRACE("Access Flagged:");
    irpass::print(ir);
  }

  irpass::constant_fold(ir);
  if (print_ir) {
    TC_TRACE("Constant folded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  global_tmps_buffer_size_ =
      std::max(irpass::offload(ir).total_size, (size_t)(8));
  if (print_ir) {
    TC_TRACE("Offloaded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::full_simplify(ir);
  if (print_ir) {
    TC_TRACE("Simplified II:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::demote_atomics(ir);
  if (print_ir) {
    TC_TRACE("Atomics demoted:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
}

FunctionType MetalCodeGen::gen(MetalRuntime *runtime) {
  // Make a copy of the name!
  const std::string taichi_kernel_name = taichi_kernel_name_;
  MetalKernelCodegen codegen(taichi_kernel_name);
  codegen.run(struct_compiled_->source_code, kernel_);
  metal::MetalKernelArgsAttributes mtl_args_attribs;
  for (const auto &arg : kernel_->args) {
    mtl_args_attribs.insert_arg(arg.dt, arg.is_nparray, arg.size,
                                arg.is_return_value);
  }
  mtl_args_attribs.finalize();
  runtime->register_taichi_kernel(
      taichi_kernel_name, codegen.kernel_source_code(),
      codegen.kernels_attribs(), global_tmps_buffer_size_, mtl_args_attribs);
  return [runtime, taichi_kernel_name](Context &ctx) {
    runtime->launch_taichi_kernel(taichi_kernel_name, &ctx);
  };
}

}  // namespace metal
TLANG_NAMESPACE_END

#endif  // TC_SUPPORTS_METAL
