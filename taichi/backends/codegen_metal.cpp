#include "codegen_metal.h"

#include <taichi/ir.h>

#ifdef TC_SUPPORTS_METAL

TLANG_NAMESPACE_BEGIN
namespace metal {
namespace {

constexpr char kKernelThreadIdName[] = "utid_";  // 'u' for unsigned
constexpr char kGlobalTmpsBufferName[] = "global_tmps_addr";
constexpr char kArgsContextName[] = "args_ctx_";

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
    args_attribs_ = MetalKernelArgsAttributes();
    for (int i = 0; i < kernel->args.size(); ++i) {
      const auto &a = kernel->args[i];
      args_attribs_.insert_arg(a.dt, a.is_nparray, a.size, a.is_return_value);
    }
    args_attribs_.finalize();

    if (args_attribs_.has_args()) {
      const auto class_name = kernel_args_classname();
      emit("namespace {{");
      emit("class {} {{", class_name);
      emit(" public:");
      push_indent();
      emit("explicit {}(device byte* addr) : addr_(addr) {{}}", class_name);
      for (const auto &arg : args_attribs_.args()) {
        const auto dt_name = metal_data_type_name(arg.dt);
        emit("device {}* arg{}() {{", dt_name, arg.index);
        if (arg.is_array) {
          emit("  // array, size={} B", arg.stride);
        } else {
          emit("  // scalar, size={} B", arg.stride);
        }
        emit("  return (device {}*)(addr_ + {});", dt_name, arg.offset_in_mem);
        emit("}}");
      }
      emit("");
      emit("int32_t extra_arg(int i, int j) {{");
      emit("  device int32_t* base = (device int32_t*)(addr_ + {});",
           args_attribs_.args_bytes());
      emit("  return *(base + (i * {}) + j);", taichi_max_num_indices);
      emit("}}");
      pop_indent();
      emit(" private:");
      emit("  device byte* addr_;");
      emit("}};");
      emit("}}  // namespace");
      emit("");
    }
  }

  void generate_serial_kernel(OffloadedStmt *stmt) {
    TC_ASSERT(stmt->task_type == OffloadedStmt::TaskType::serial);
    const std::string mtl_kernel_name = make_kernel_name();
    emit_mtl_kernel_func_sig(mtl_kernel_name);
    emit("  // serial");
    emit("  if ({} > 0) return;", kKernelThreadIdName);

    MetalKernelAttributes ka;
    ka.name = mtl_kernel_name;
    ka.task_type = stmt->task_type;
    ka.num_threads = 1;

    current_kernel_attribs_ = &ka;
    stmt->body->accept(this);
    emit("}}\n");
    current_kernel_attribs_ = nullptr;

    mtl_kernels_attribs_.push_back(ka);
  }

  void generate_range_for_kernel(OffloadedStmt *stmt) {
    TC_ASSERT(stmt->task_type == OffloadedStmt::TaskType::range_for);
    const std::string mtl_kernel_name = make_kernel_name();
    emit_mtl_kernel_func_sig(mtl_kernel_name);

    MetalKernelAttributes ka;
    ka.name = mtl_kernel_name;
    ka.task_type = stmt->task_type;

    auto &range_for_attribs = ka.range_for_attribs;
    range_for_attribs.const_begin = stmt->const_begin;
    range_for_attribs.const_end = stmt->const_end;
    range_for_attribs.begin =
        (stmt->const_begin ? stmt->begin_value : stmt->begin_offset);
    range_for_attribs.end =
        (stmt->const_end ? stmt->end_value : stmt->end_offset);

    push_indent();
    if (range_for_attribs.const_range()) {
      ka.num_threads = range_for_attribs.end - range_for_attribs.begin;
      emit("// range_for, range known at compile time");
      emit("if ({} >= {}) return;", kKernelThreadIdName, ka.num_threads);
    } else {
      ka.num_threads = -1;
      emit("// range_for, range known at runtime");
      emit("{{");
      push_indent();
      const auto begin_stmt = stmt->const_begin
                                  ? std::to_string(stmt->begin_value)
                                  : inject_load_global_tmp(stmt->begin_offset);
      const auto end_stmt = stmt->const_end
                                ? std::to_string(stmt->end_value)
                                : inject_load_global_tmp(stmt->end_offset);
      emit("if ({} >= ({} - {})) return;", kKernelThreadIdName, end_stmt,
           begin_stmt);
      pop_indent();
      emit("}}");
    }
    pop_indent();

    current_kernel_attribs_ = &ka;
    stmt->body->accept(this);
    emit("}}\n");
    current_kernel_attribs_ = nullptr;

    mtl_kernels_attribs_.push_back(ka);
  }

  std::string inject_load_global_tmp(int offset, DataType dt = DataType::i32) {
    const auto vt = VectorType(/*width=*/1, dt);
    auto gtmp = Stmt::make<GlobalTemporaryStmt>(offset, vt);
    gtmp->accept(this);
    auto gload = Stmt::make<GlobalLoadStmt>(gtmp.get());
    gload->ret_type = vt;
    gload->accept(this);
    return gload->raw_name();
  }

  std::string make_kernel_name() {
    return fmt::format("{}_{}", mtl_kernel_prefix_, mtl_kernel_count_++);
  }

  inline std::string kernel_args_classname() const {
    return fmt::format("{}_args", mtl_kernel_prefix_);
  }

  void emit_mtl_kernel_func_sig(const std::string &kernel_name) {
    emit("kernel void {}(", kernel_name);
    emit("    device byte* addr [[buffer(0)]],");
    emit("    device byte* {} [[buffer(1)]],", kGlobalTmpsBufferName);
    if (args_attribs_.has_args()) {
      emit("    device byte* args_addr [[buffer(2)]],");
    }
    emit("    const uint {} [[thread_position_in_grid]]) {{",
         kKernelThreadIdName);
    if (args_attribs_.has_args()) {
      emit("  {} {}(args_addr);", kernel_args_classname(), kArgsContextName);
    }
  }

  void push_indent() {
    indent_ += "  ";
  }

  void pop_indent() {
    indent_.pop_back();
    indent_.pop_back();
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
