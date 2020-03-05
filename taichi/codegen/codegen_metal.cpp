#include "codegen_metal.h"

#include <string>
#include <taichi/ir/ir.h>

TLANG_NAMESPACE_BEGIN
namespace metal {
namespace {

constexpr char kKernelThreadIdName[] = "utid_";  // 'u' for unsigned
constexpr char kRootBufferName[] = "root_addr";
constexpr char kGlobalTmpsBufferName[] = "global_tmps_addr";
constexpr char kArgsBufferName[] = "args_addr";
constexpr char kArgsContextName[] = "args_ctx_";

class MetalKernelCodegen : public IRVisitor {
 public:
  MetalKernelCodegen(const std::string &mtl_kernel_prefix,
                     const std::string &root_snode_type_name,
                     Kernel *kernel,
                     const StructCompiledResult *compiled_snode_structs)
      : mtl_kernel_prefix_(mtl_kernel_prefix),
        root_snode_type_name_(root_snode_type_name),
        kernel_(kernel),
        compiled_snode_structs_(compiled_snode_structs),
        needs_root_buffer_(compiled_snode_structs_->root_size > 0),
        args_attribs_(kernel_->args) {
    // allow_undefined_visitor = true;
  }

  const MetalKernelArgsAttributes &kernel_args_attribs() const {
    return args_attribs_;
  }

  const std::string &kernel_source_code() const {
    return kernel_src_code_;
  }

  const std::vector<MetalKernelAttributes> &kernels_attribs() const {
    return mtl_kernels_attribs_;
  }

  void run() {
    generate_mtl_header(compiled_snode_structs_->source_code);
    generate_kernel_args_struct();
    kernel_->ir->accept(this);
  }

  void visit(Block *stmt) override {
    if (!is_top_level_) {
      push_indent();
    }
    for (auto &s : stmt->statements) {
      s->accept(this);
    }
    if (!is_top_level_) {
      pop_indent();
    }
  }

  void visit(AllocaStmt *alloca) override {
    emit(R"({} {}(0);)", metal_data_type_name(alloca->element_type()),
         alloca->raw_name());
  }

  void visit(ConstStmt *const_stmt) override {
    TI_ASSERT(const_stmt->width() == 1);
    emit("const {} {} = {};", metal_data_type_name(const_stmt->element_type()),
         const_stmt->raw_name(), const_stmt->val[0].stringify());
  }

  void visit(LocalLoadStmt *stmt) override {
    // TODO: optimize for partially vectorized load...
    bool linear_index = true;
    for (int i = 0; i < (int)stmt->ptr.size(); i++) {
      if (stmt->ptr[i].offset != i) {
        linear_index = false;
      }
    }
    if (stmt->same_source() && linear_index &&
        stmt->width() == stmt->ptr[0].var->width()) {
      auto ptr = stmt->ptr[0].var;
      emit("const {} {}({});", metal_data_type_name(stmt->element_type()),
           stmt->raw_name(), ptr->raw_name());
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    emit(R"({} = {};)", stmt->ptr->raw_name(), stmt->data->raw_name());
  }

  void visit(GetRootStmt *stmt) override {
    // Should we assert |root_stmt_| is assigned only once?
    TI_ASSERT(needs_root_buffer_);
    root_stmt_ = stmt;
    emit(R"({} {}({});)", root_snode_type_name_, stmt->raw_name(),
         kRootBufferName);
  }

  void visit(GetChStmt *stmt) override {
    if (stmt->output_snode->is_place()) {
      emit(R"(device {}* {} = {}.get{}().val;)",
           metal_data_type_name(stmt->output_snode->dt), stmt->raw_name(),
           stmt->input_ptr->raw_name(), stmt->chid);
    } else {
      emit(R"({} {} = {}.get{}();)", stmt->output_snode->node_type_name,
           stmt->raw_name(), stmt->input_ptr->raw_name(), stmt->chid);
    }
  }

  void visit(LinearizeStmt *stmt) override {
    std::string val = "0";
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      val = fmt::format("({} * {} + {})", val, stmt->strides[i],
                        stmt->inputs[i]->raw_name());
    }
    emit(R"(auto {} = {};)", stmt->raw_name(), val);
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override {
    emit(R"(auto {} = ((({} + {}) >> {}) & ((1 << {}) - 1));)",
         stmt->raw_name(), stmt->offset, stmt->input->raw_name(),
         stmt->bit_begin, stmt->bit_end - stmt->bit_begin);
  }

  void visit(SNodeLookupStmt *stmt) override {
    std::string parent;
    if (stmt->input_snode) {
      parent = stmt->input_snode->raw_name();
    } else {
      TI_ASSERT(root_stmt_ != nullptr);
      parent = root_stmt_->raw_name();
    }

    emit(R"({}_ch {} = {}.children({});)", stmt->snode->node_type_name,
         stmt->raw_name(), parent, stmt->input_index->raw_name());
  }

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit(R"(*{} = {};)", stmt->ptr->raw_name(), stmt->data->raw_name());
  }

  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit(R"({} {} = *{};)", metal_data_type_name(stmt->element_type()),
         stmt->raw_name(), stmt->ptr->raw_name());
  }

  void visit(ArgLoadStmt *stmt) override {
    const auto dt = metal_data_type_name(stmt->element_type());
    if (stmt->is_ptr) {
      emit("device {} *{} = {}.arg{}();", dt, stmt->raw_name(),
           kArgsContextName, stmt->arg_id);
    } else {
      emit("const {} {} = *{}.arg{}();", dt, stmt->raw_name(), kArgsContextName,
           stmt->arg_id);
    }
  }

  void visit(ArgStoreStmt *stmt) override {
    const auto dt = metal_data_type_name(stmt->element_type());
    TI_ASSERT(!stmt->is_ptr);
    emit("*{}.arg{}() = {};", kArgsContextName, stmt->arg_id,
         stmt->val->raw_name());
  }

  void visit(ExternalPtrStmt *stmt) override {
    // Used mostly for transferring data between host (e.g. numpy array) and
    // Metal.
    TI_ASSERT(stmt->width() == 1);
    const auto linear_index_name =
        fmt::format("{}_linear_index_", stmt->raw_name());
    emit("int {} = 0;", linear_index_name);
    emit("{{");
    push_indent();
    const auto *argload = stmt->base_ptrs[0]->as<ArgLoadStmt>();
    const int arg_id = argload->arg_id;
    const int num_indices = stmt->indices.size();
    std::vector<std::string> size_var_names;
    for (int i = 0; i < num_indices; i++) {
      std::string var_name = fmt::format("{}_size{}_", stmt->raw_name(), i);
      emit("const int {} = {}.extra_arg({}, {});", var_name, kArgsContextName,
           arg_id, i);
      size_var_names.push_back(std::move(var_name));
    }
    for (int i = 0; i < num_indices; i++) {
      emit("{} *= {};", linear_index_name, size_var_names[i]);
      emit("{} += {};", linear_index_name, stmt->indices[i]->raw_name());
    }

    pop_indent();
    emit("}}");

    const auto dt = metal_data_type_name(stmt->element_type());
    emit("device {} *{} = ({} + {});", dt, stmt->raw_name(),
         stmt->base_ptrs[0]->raw_name(), linear_index_name);
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    const auto dt = metal_data_type_name(stmt->element_type());
    emit("device {}* {} = reinterpret_cast<device {}*>({} + {});", dt,
         stmt->raw_name(), dt, kGlobalTmpsBufferName, stmt->offset);
  }

  void visit(LoopIndexStmt *stmt) override {
    TI_ASSERT(current_kernel_attribs_->task_type ==
              OffloadedStmt::TaskType::range_for);
    TI_ASSERT(!stmt->is_struct_for && stmt->index == 0);
    if (current_kernel_attribs_->range_for_attribs.const_begin) {
      emit("const int {} = (static_cast<int>({}) + {});", stmt->raw_name(),
           kKernelThreadIdName,
           current_kernel_attribs_->range_for_attribs.begin);
    } else {
      auto begin_stmt = inject_load_global_tmp(
          current_kernel_attribs_->range_for_attribs.begin);
      emit("const int {} = (static_cast<int>({}) + {});", stmt->raw_name(),
           kKernelThreadIdName, begin_stmt);
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->op_type != UnaryOpType::cast) {
      emit("const {} {} = {}({});", metal_data_type_name(stmt->element_type()),
           stmt->raw_name(), metal_unary_op_type_symbol(stmt->op_type),
           stmt->operand->raw_name());
    } else {
      // cast
      if (stmt->cast_by_value) {
        emit("const {} {} = static_cast<{}>({});",
             metal_data_type_name(stmt->element_type()), stmt->raw_name(),
             metal_data_type_name(stmt->cast_type), stmt->operand->raw_name());
      } else {
        // reinterpret the bit pattern
        const auto to_type = to_metal_type(stmt->cast_type);
        const auto to_type_name = metal_data_type_name(to_type);
        TI_ASSERT(metal_data_type_bytes(
                      to_metal_type(stmt->operand->element_type())) ==
                  metal_data_type_bytes(to_type));
        emit("const {} {} = union_cast<{}>({});", to_type_name,
             stmt->raw_name(), to_type_name, stmt->operand->raw_name());
      }
    }
  }

  void visit(BinaryOpStmt *bin) override {
    const auto dt_name = metal_data_type_name(bin->element_type());
    const auto lhs_name = bin->lhs->raw_name();
    const auto rhs_name = bin->rhs->raw_name();
    const auto bin_name = bin->raw_name();
    if (bin->op_type == BinaryOpType::floordiv) {
      if (is_integral(bin->element_type())) {
        emit("const {} {} = ifloordiv({}, {});", dt_name, bin_name, lhs_name,
             rhs_name);
      } else {
        emit("const {} {} = floor({} / {});", dt_name, bin_name, lhs_name,
             rhs_name);
      }
      return;
    }
    const auto binop = metal_binary_op_type_symbol(bin->op_type);
    if (is_metal_binary_op_infix(bin->op_type)) {
      emit("const {} {} = ({} {} {});", dt_name, bin_name, lhs_name, binop,
           rhs_name);
    } else {
      // This is a function call
      emit("const {} {} =  {}({}, {});", dt_name, bin_name, binop, lhs_name,
           rhs_name);
    }
  }

  void visit(TernaryOpStmt *tri) override {
    TI_ASSERT(tri->op_type == TernaryOpType::select);
    emit("const {} {} = ({}) ? ({}) : ({});",
         metal_data_type_name(tri->element_type()), tri->raw_name(),
         tri->op1->raw_name(), tri->op2->raw_name(), tri->op3->raw_name());
  }

  void visit(AtomicOpStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    TI_ASSERT(stmt->op_type == AtomicOpType::add);
    const auto dt = stmt->val->element_type();
    if (dt == DataType::i32) {
      emit(
          "const auto {} = atomic_fetch_add_explicit((device atomic_int*){}, "
          "{}, "
          "metal::memory_order_relaxed);",
          stmt->raw_name(), stmt->dest->raw_name(), stmt->val->raw_name());
    } else if (dt == DataType::f32) {
      emit("const float {} = fatomic_fetch_add({}, {});", stmt->raw_name(),
           stmt->dest->raw_name(), stmt->val->raw_name());
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(IfStmt *if_stmt) override {
    emit("if ({}) {{", if_stmt->cond->raw_name());
    if (if_stmt->true_statements) {
      if_stmt->true_statements->accept(this);
    }
    emit("}} else {{");
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
    emit("}}");
  }

  void visit(RangeForStmt *for_stmt) override {
    TI_ASSERT(for_stmt->width() == 1);
    auto *loop_var = for_stmt->loop_var;
    if (loop_var->ret_type.data_type == DataType::i32) {
      if (!for_stmt->reversed) {
        emit("for (int {}_ = {}; {}_ < {}; {}_ = {}_ + {}) {{",
             loop_var->raw_name(), for_stmt->begin->raw_name(),
             loop_var->raw_name(), for_stmt->end->raw_name(),
             loop_var->raw_name(), loop_var->raw_name(), 1);
        emit("  int {} = {}_;", loop_var->raw_name(), loop_var->raw_name());
      } else {
        // reversed for loop
        emit("for (int {}_ = {} - 1; {}_ >= {}; {}_ = {}_ - {}) {{",
             loop_var->raw_name(), for_stmt->end->raw_name(),
             loop_var->raw_name(), for_stmt->begin->raw_name(),
             loop_var->raw_name(), loop_var->raw_name(), 1);
        emit("  int {} = {}_;", loop_var->raw_name(), loop_var->raw_name());
      }
    } else {
      TI_ASSERT(!for_stmt->reversed);
      const auto type_name = metal_data_type_name(loop_var->element_type());
      emit("for ({} {} = {}; {} < {}; {} = {} + ({})1) {{", type_name,
           loop_var->raw_name(), for_stmt->begin->raw_name(),
           loop_var->raw_name(), for_stmt->end->raw_name(),
           loop_var->raw_name(), loop_var->raw_name(), type_name);
    }
    for_stmt->body->accept(this);
    emit("}}");
  }

  void visit(StructForStmt *) override {
    TI_ERROR("Struct for cannot be nested.");
  }

  void visit(OffloadedStmt *stmt) override {
    TI_ASSERT(is_top_level_);
    is_top_level_ = false;
    using Type = OffloadedStmt::TaskType;
    if (stmt->task_type == Type::serial) {
      generate_serial_kernel(stmt);
    } else if (stmt->task_type == Type::range_for) {
      generate_range_for_kernel(stmt);
    } else {
      // struct_for is automatically lowered to ranged_for for dense snodes
      // (#378). So we only need to support serial and range_for tasks.
      TI_ERROR("Unsupported offload type={} on Metal arch", stmt->task_name());
    }
    is_top_level_ = true;
  }

  void visit(WhileControlStmt *stmt) override {
    emit("if (!{}) break;", stmt->cond->raw_name());
  }

  void visit(WhileStmt *stmt) override {
    emit("while (true) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(RandStmt *stmt) override {
    TI_ERROR("Metal arch doesn't support ti.random() yet");
  }

  void visit(PrintStmt *stmt) override {
    // TODO: Add a flag to control whether ignoring print() stmt is allowed.
    TI_WARN("Cannot print inside Metal kernel, ignored");
  }

 private:
  void generate_mtl_header(const std::string &snode_structs_source_code) {
    emit("#include <metal_stdlib>");
    emit("using namespace metal;");
    emit("");
    emit("namespace {{");
    emit("");
    generate_common_functions();
    kernel_src_code_ += snode_structs_source_code;
    emit("}}  // namespace");
    emit("");
  }

  void generate_common_functions() {
#define TI_INSIDE_METAL_CODEGEN
#include <taichi/platform/metal/helpers.metal.h>
    kernel_src_code_ += kMetalHelpersSourceCode;
#undef TI_INSIDE_METAL_CODEGEN
    emit("\n");
  }

  void generate_kernel_args_struct() {
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
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::serial);
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
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::range_for);
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
    int buffer_idx = 0;
    if (needs_root_buffer_) {
      emit("    device byte* {} [[buffer({})]],", kRootBufferName,
           buffer_idx++);
    }
    emit("    device byte* {} [[buffer({})]],", kGlobalTmpsBufferName,
         buffer_idx++);
    if (args_attribs_.has_args()) {
      emit("    device byte* {} [[buffer({})]],", kArgsBufferName,
           buffer_idx++);
    }
    emit("    const uint {} [[thread_position_in_grid]]) {{",
         kKernelThreadIdName);
    if (args_attribs_.has_args()) {
      emit("  {} {}({});", kernel_args_classname(), kArgsContextName,
           kArgsBufferName);
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
  const std::string root_snode_type_name_;
  Kernel *const kernel_;
  const StructCompiledResult *const compiled_snode_structs_;
  const bool needs_root_buffer_;
  const MetalKernelArgsAttributes args_attribs_;

  bool is_top_level_{true};
  int mtl_kernel_count_{0};
  std::vector<MetalKernelAttributes> mtl_kernels_attribs_;
  GetRootStmt *root_stmt_{nullptr};
  MetalKernelAttributes *current_kernel_attribs_{nullptr};
  std::string kernel_src_code_;
  std::string indent_;
};

}  // namespace

MetalCodeGen::MetalCodeGen(const std::string &kernel_name,
                           const StructCompiledResult *struct_compiled)
    : id_(Program::get_kernel_id()),
      taichi_kernel_name_(fmt::format("mtl_k{:04d}_{}", id_, kernel_name)),
      struct_compiled_(struct_compiled) {
}

FunctionType MetalCodeGen::compile(Program &,
                                   Kernel &kernel,
                                   MetalRuntime *runtime) {
  this->prog_ = &kernel.program;
  this->kernel_ = &kernel;
  lower();
  return gen(*prog_->snode_root, runtime);
}

void MetalCodeGen::lower() {
  auto ir = kernel_->ir;
  const bool print_ir = prog_->config.print_ir;
  if (print_ir) {
    TI_TRACE("Initial IR:");
    irpass::print(ir);
  }

  if (kernel_->grad) {
    irpass::reverse_segments(ir);
    irpass::re_id(ir);
    if (print_ir) {
      TI_TRACE("Segment reversed (for autodiff):");
      irpass::print(ir);
    }
  }

  irpass::lower(ir);
  irpass::re_id(ir);
  if (print_ir) {
    TI_TRACE("Lowered:");
    irpass::print(ir);
  }

  irpass::typecheck(ir);
  irpass::re_id(ir);
  if (print_ir) {
    TI_TRACE("Typechecked:");
    irpass::print(ir);
  }

  irpass::demote_dense_struct_fors(ir);
  irpass::typecheck(ir);
  if (print_ir) {
    TI_TRACE("Dense Struct-for demoted:");
    irpass::print(ir);
  }

  irpass::constant_fold(ir);
  if (prog_->config.simplify_before_lower_access) {
    irpass::simplify(ir);
    irpass::re_id(ir);
    if (print_ir) {
      TI_TRACE("Simplified I:");
      irpass::print(ir);
    }
  }

  if (kernel_->grad) {
    irpass::demote_atomics(ir);
    irpass::full_simplify(ir, prog_->config);
    irpass::typecheck(ir);
    if (print_ir) {
      TI_TRACE("Before make_adjoint:");
      irpass::print(ir);
    }
    irpass::make_adjoint(ir);
    if (print_ir) {
      TI_TRACE("After make_adjoint:");
      irpass::print(ir);
    }
    irpass::typecheck(ir);
  }

  irpass::lower_access(ir, prog_->config.use_llvm);
  irpass::re_id(ir);
  if (print_ir) {
    TI_TRACE("Access Lowered:");
    irpass::print(ir);
  }

  irpass::die(ir);
  irpass::re_id(ir);
  if (print_ir) {
    TI_TRACE("DIEd:");
    irpass::print(ir);
  }

  irpass::flag_access(ir);
  irpass::re_id(ir);
  if (print_ir) {
    TI_TRACE("Access Flagged:");
    irpass::print(ir);
  }

  irpass::constant_fold(ir);
  if (print_ir) {
    TI_TRACE("Constant folded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  global_tmps_buffer_size_ =
      std::max(irpass::offload(ir).total_size, (size_t)(1));
  if (print_ir) {
    TI_TRACE("Offloaded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::full_simplify(ir, prog_->config);
  if (print_ir) {
    TI_TRACE("Simplified II:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::demote_atomics(ir);
  if (print_ir) {
    TI_TRACE("Atomics demoted:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
}

FunctionType MetalCodeGen::gen(const SNode &root_snode, MetalRuntime *runtime) {
  // Make a copy of the name!
  const std::string taichi_kernel_name = taichi_kernel_name_;
  MetalKernelCodegen codegen(taichi_kernel_name, root_snode.node_type_name,
                             kernel_, struct_compiled_);
  codegen.run();
  runtime->register_taichi_kernel(
      taichi_kernel_name, codegen.kernel_source_code(),
      codegen.kernels_attribs(), global_tmps_buffer_size_,
      codegen.kernel_args_attribs());
  return [runtime, taichi_kernel_name](Context &ctx) {
    runtime->launch_taichi_kernel(taichi_kernel_name, &ctx);
  };
}

}  // namespace metal
TLANG_NAMESPACE_END
