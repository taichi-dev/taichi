#include "codegen_metal.h"

#include <functional>
#include <string>

#include "taichi/backends/metal/constants.h"
#include "taichi/backends/metal/features.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/util/line_appender.h"
#include "taichi/math/arithmetic.h"
#include "taichi/backends/metal/api.h"

TLANG_NAMESPACE_BEGIN
namespace metal {
namespace {

namespace shaders {
#define TI_INSIDE_METAL_CODEGEN
#include "taichi/backends/metal/shaders/ad_stack.metal.h"
#include "taichi/backends/metal/shaders/helpers.metal.h"
#include "taichi/backends/metal/shaders/print.metal.h"
#include "taichi/backends/metal/shaders/runtime_kernels.metal.h"
#undef TI_INSIDE_METAL_CODEGEN

#include "taichi/backends/metal/shaders/print.metal.h"
#include "taichi/backends/metal/shaders/runtime_structs.metal.h"

}  // namespace shaders

using BuffersEnum = KernelAttributes::Buffers;

constexpr char kKernelThreadIdName[] = "utid_";        // 'u' for unsigned
constexpr char kKernelGridSizeName[] = "ugrid_size_";  // 'u' for unsigned
constexpr char kKernelTidInSimdgroupName[] = "utid_in_simdg_";
constexpr char kRootBufferName[] = "root_addr";
constexpr char kGlobalTmpsBufferName[] = "global_tmps_addr";
constexpr char kContextBufferName[] = "ctx_addr";
constexpr char kContextVarName[] = "kernel_ctx_";
constexpr char kRuntimeBufferName[] = "runtime_addr";
constexpr char kRuntimeVarName[] = "runtime_";
constexpr char kPrintBufferName[] = "print_addr";
constexpr char kPrintAllocVarName[] = "print_alloc_";
constexpr char kLinearLoopIndexName[] = "linear_loop_idx_";
constexpr char kListgenElemVarName[] = "listgen_elem_";
constexpr char kRandStateVarName[] = "rand_state_";
constexpr char kMemAllocVarName[] = "mem_alloc_";
constexpr char kTlsBufferName[] = "tls_buffer_";

std::string buffer_to_name(BuffersEnum b) {
  switch (b) {
    case BuffersEnum::Root:
      return kRootBufferName;
    case BuffersEnum::GlobalTmps:
      return kGlobalTmpsBufferName;
    case BuffersEnum::Context:
      return kContextBufferName;
    case BuffersEnum::Runtime:
      return kRuntimeBufferName;
    case BuffersEnum::Print:
      return kPrintBufferName;
    default:
      TI_NOT_IMPLEMENTED;
      break;
  }
  return {};
}

class KernelCodegen : public IRVisitor {
 private:
  enum class Section {
    Headers,
    Structs,
    KernelFuncs,
    Kernels,
  };

  static constexpr Section kAllSections[] = {
      Section::Headers,
      Section::Structs,
      Section::KernelFuncs,
      Section::Kernels,
  };

 public:
  // TODO(k-ye): Create a Params to hold these ctor params.
  KernelCodegen(const std::string &taichi_kernel_name,
                const std::string &root_snode_type_name,
                Kernel *kernel,
                const CompiledStructs *compiled_structs,
                PrintStringTable *print_strtab,
                const CodeGen::Config &config)
      : mtl_kernel_prefix_(taichi_kernel_name),
        root_snode_type_name_(root_snode_type_name),
        kernel_(kernel),
        compiled_structs_(compiled_structs),
        needs_root_buffer_(compiled_structs_->root_size > 0),
        ctx_attribs_(*kernel_),
        print_strtab_(print_strtab),
        cgen_config_(config) {
    ti_kernel_attribus_.name = taichi_kernel_name;
    ti_kernel_attribus_.is_jit_evaluator = kernel->is_evaluator;
    // allow_undefined_visitor = true;
    for (const auto s : kAllSections) {
      section_appenders_[s] = LineAppender();
    }
  }

  const KernelContextAttributes &kernel_ctx_attribs() const {
    return ctx_attribs_;
  }

  const TaichiKernelAttributes &ti_kernels_attribs() const {
    return ti_kernel_attribus_;
  }

  std::string run() {
    emit_headers();
    generate_structs();
    generate_kernels();

    std::string source_code;
    for (const auto s : kAllSections) {
      source_code += section_appenders_.find(s)->second.lines();
      source_code += '\n';
    }
    return source_code;
  }

  void visit(Block *stmt) override {
    if (!is_top_level_) {
      current_appender().push_indent();
    }
    for (auto &s : stmt->statements) {
      s->accept(this);
    }
    if (!is_top_level_) {
      current_appender().pop_indent();
    }
  }

  void visit(AllocaStmt *alloca) override {
    emit(R"({} {}(0);)", metal_data_type_name(alloca->element_type()),
         alloca->raw_name());
  }

  void visit(ConstStmt *const_stmt) override {
    TI_ASSERT(const_stmt->width() == 1);
    emit("constexpr {} {} = {};",
         metal_data_type_name(const_stmt->element_type()),
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
    // E.g. `parent.get*(runtime, mem_alloc)`
    const auto get_call =
        fmt::format("{}.get{}({}, {})", stmt->input_ptr->raw_name(), stmt->chid,
                    kRuntimeVarName, kMemAllocVarName);
    if (stmt->output_snode->is_place()) {
      emit(R"(device {}* {} = {}.val;)",
           metal_data_type_name(stmt->output_snode->dt), stmt->raw_name(),
           get_call);
    } else {
      emit(R"({} {} = {};)", stmt->output_snode->node_type_name,
           stmt->raw_name(), get_call);
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

  void visit(BitExtractStmt *stmt) override {
    emit(R"(auto {} = (({} >> {}) & ((1 << {}) - 1));)", stmt->raw_name(),
         stmt->input->raw_name(), stmt->bit_begin,
         stmt->bit_end - stmt->bit_begin);
  }

  void visit(SNodeLookupStmt *stmt) override {
    std::string parent;
    if (stmt->input_snode) {
      parent = stmt->input_snode->raw_name();
    } else {
      TI_ASSERT(root_stmt_ != nullptr);
      parent = root_stmt_->raw_name();
    }
    const auto *sn = stmt->snode;
    const auto snty = sn->type;
    const std::string index_name = stmt->input_index->raw_name();

    if (stmt->activate) {
      TI_ASSERT(is_supported_sparse_type(snty));
      emit("{}.activate({});", parent, index_name);
    }
    emit(R"({}_ch {} = {}.children({});)", sn->node_type_name, stmt->raw_name(),
         parent, index_name);
  }

  void visit(SNodeOpStmt *stmt) override {
    const std::string result_var = stmt->raw_name();
    const auto opty = stmt->op_type;
    if (opty == SNodeOpType::is_active || opty == SNodeOpType::append ||
        opty == SNodeOpType::length) {
      emit("int {};", result_var);
    }
    emit("{{");
    {
      ScopedIndent s(current_appender());
      const auto &parent = stmt->ptr->raw_name();
      const bool is_dynamic = (stmt->snode->type == SNodeType::dynamic);
      std::string ch_id;
      if (is_dynamic &&
          (opty == SNodeOpType::deactivate || opty == SNodeOpType::append ||
           opty == SNodeOpType::length)) {
        // For these ops, `dynamic` is a special case because |stmt| doesn't
        // contain an index to its cells. Setting it to zero to store the
        // address of the first child into |ch_addr|.
        ch_id = "0";
      } else {
        ch_id = stmt->val->raw_name();
      }
      const std::string ch_addr =
          fmt::format("{}.children({}).addr()", stmt->ptr->raw_name(), ch_id);
      if (opty == SNodeOpType::is_active) {
        emit("{} = {}.is_active({});", result_var, parent,
             stmt->val->raw_name());
      } else if (opty == SNodeOpType::activate) {
        emit("{}.activate({});", parent, stmt->val->raw_name());
      } else if (opty == SNodeOpType::deactivate) {
        if (is_dynamic) {
          emit("{}.deactivate();", parent);
        } else {
          emit("{}.deactivate({});", parent, stmt->val->raw_name());
        }
      } else if (opty == SNodeOpType::append) {
        TI_ASSERT(is_dynamic);
        TI_ASSERT(stmt->ret_type.data_type == DataType::i32);
        emit("{} = {}.append({});", result_var, parent, stmt->val->raw_name());
      } else if (opty == SNodeOpType::length) {
        TI_ASSERT(is_dynamic);
        emit("{} = {}.length();", result_var, parent);
      } else {
        TI_NOT_IMPLEMENTED
      }
    }
    emit("}}");
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
      emit("device {} *{} = {}.arg{}();", dt, stmt->raw_name(), kContextVarName,
           stmt->arg_id);
    } else {
      emit("const {} {} = *{}.arg{}();", dt, stmt->raw_name(), kContextVarName,
           stmt->arg_id);
    }
  }

  void visit(KernelReturnStmt *stmt) override {
    // TODO: use stmt->ret_id instead of 0 as index
    emit("*{}.ret0() = {};", kContextVarName, stmt->value->raw_name());
  }

  void visit(ExternalPtrStmt *stmt) override {
    // Used mostly for transferring data between host (e.g. numpy array) and
    // Metal.
    TI_ASSERT(stmt->width() == 1);
    const auto linear_index_name =
        fmt::format("{}_linear_index_", stmt->raw_name());
    emit("int {} = 0;", linear_index_name);
    emit("{{");
    {
      ScopedIndent s(current_appender());
      const auto *argload = stmt->base_ptrs[0]->as<ArgLoadStmt>();
      const int arg_id = argload->arg_id;
      const int num_indices = stmt->indices.size();
      std::vector<std::string> size_var_names;
      for (int i = 0; i < num_indices; i++) {
        std::string var_name = fmt::format("{}_size{}_", stmt->raw_name(), i);
        emit("const int {} = {}.extra_arg({}, {});", var_name, kContextVarName,
             arg_id, i);
        size_var_names.push_back(std::move(var_name));
      }
      for (int i = 0; i < num_indices; i++) {
        emit("{} *= {};", linear_index_name, size_var_names[i]);
        emit("{} += {};", linear_index_name, stmt->indices[i]->raw_name());
      }
    }
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

  void visit(ThreadLocalPtrStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("thread auto* {} = reinterpret_cast<thread {}*>({} + {});",
         stmt->raw_name(), metal_data_type_name(stmt->element_type()),
         kTlsBufferName, stmt->offset);
  }

  void visit(LoopIndexStmt *stmt) override {
    const auto stmt_name = stmt->raw_name();
    if (stmt->loop->is<OffloadedStmt>()) {
      using TaskType = OffloadedStmt::TaskType;
      const auto type = stmt->loop->as<OffloadedStmt>()->task_type;
      if (type == TaskType::range_for) {
        TI_ASSERT(stmt->index == 0);
        emit("const int {} = {};", stmt_name, kLinearLoopIndexName);
      } else if (type == TaskType::struct_for) {
        emit("const int {} = {}.coords[{}];", stmt_name, kListgenElemVarName,
             stmt->index);
      } else {
        TI_NOT_IMPLEMENTED;
      }
    } else if (stmt->loop->is<RangeForStmt>()) {
      TI_ASSERT(stmt->index == 0);
      emit("const int {} = {};", stmt_name, stmt->loop->raw_name());
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->op_type == UnaryOpType::cast_value) {
      emit("const {} {} = static_cast<{}>({});",
           metal_data_type_name(stmt->element_type()), stmt->raw_name(),
           metal_data_type_name(stmt->cast_type), stmt->operand->raw_name());
    } else if (stmt->op_type == UnaryOpType::cast_bits) {
      // reinterpret the bit pattern
      const auto to_type = to_metal_type(stmt->cast_type);
      const auto to_type_name = metal_data_type_name(to_type);
      TI_ASSERT(
          metal_data_type_bytes(to_metal_type(stmt->operand->element_type())) ==
          metal_data_type_bytes(to_type));
      emit("const {} {} = union_cast<{}>({});", to_type_name, stmt->raw_name(),
           to_type_name, stmt->operand->raw_name());
    } else {
      emit("const {} {} = {}({});", metal_data_type_name(stmt->element_type()),
           stmt->raw_name(), metal_unary_op_type_symbol(stmt->op_type),
           stmt->operand->raw_name());
    }
  }

  void visit(BinaryOpStmt *bin) override {
    const auto dt_name = metal_data_type_name(bin->element_type());
    const auto lhs_name = bin->lhs->raw_name();
    const auto rhs_name = bin->rhs->raw_name();
    const auto bin_name = bin->raw_name();
    const auto op_type = bin->op_type;
    if (op_type == BinaryOpType::floordiv) {
      if (is_integral(bin->ret_type.data_type)) {
        emit("const {} {} = ifloordiv({}, {});", dt_name, bin_name, lhs_name,
             rhs_name);
      } else {
        emit("const {} {} = floor({} / {});", dt_name, bin_name, lhs_name,
             rhs_name);
      }
      return;
    }
    if (op_type == BinaryOpType::pow && is_integral(bin->ret_type.data_type)) {
      // TODO(k-ye): Make sure the type is not i64?
      emit("const {} {} = pow_i32({}, {});", dt_name, bin_name, lhs_name,
           rhs_name);
      return;
    }
    const auto binop = metal_binary_op_type_symbol(op_type);
    if (is_metal_binary_op_infix(op_type)) {
      if (is_comparison(op_type)) {
        // TODO(#577): Taichi uses -1 as true due to LLVM i1... See
        // https://github.com/taichi-dev/taichi/blob/6989c0e21d437a9ffdc0151cee9d3aa2aaa2241d/taichi/codegen/codegen_llvm.cpp#L564
        // This is a workaround to make Metal compatible with the behavior.
        emit("const {} {} = -({} {} {});", dt_name, bin_name, lhs_name, binop,
             rhs_name);
      } else {
        emit("const {} {} = ({} {} {});", dt_name, bin_name, lhs_name, binop,
             rhs_name);
      }
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
    const auto dt = stmt->val->element_type();
    const auto op_type = stmt->op_type;
    std::string op_name;
    bool handle_float = false;
    if (op_type == AtomicOpType::add || op_type == AtomicOpType::min ||
        op_type == AtomicOpType::max) {
      op_name = atomic_op_type_name(op_type);
      handle_float = true;
    } else if (op_type == AtomicOpType::bit_and ||
               op_type == AtomicOpType::bit_or ||
               op_type == AtomicOpType::bit_xor) {
      // Skip "bit_"
      op_name = atomic_op_type_name(op_type).substr(/*pos=*/4);
      handle_float = false;
    } else {
      TI_NOT_IMPLEMENTED;
    }

    std::string val_var = stmt->val->raw_name();
    // TODO(k-ye): This is not a very reliable way to detect if we're in TLS
    // xlogues...
    const bool is_tls_reduction =
        (inside_tls_epilogue_ && (op_type == AtomicOpType::add));
    const bool use_simd_in_tls_reduction =
        (is_tls_reduction && cgen_config_.allow_simdgroup);
    if (use_simd_in_tls_reduction) {
      val_var += "_simd_val_";
      emit("const auto {} = simd_sum({});", val_var, stmt->val->raw_name());
      emit("if ({} == 0) {{", kKernelTidInSimdgroupName);
      current_appender().push_indent();
    }

    if (dt == DataType::i32) {
      emit(
          "const auto {} = atomic_fetch_{}_explicit((device atomic_int*){}, "
          "{}, "
          "metal::memory_order_relaxed);",
          stmt->raw_name(), op_name, stmt->dest->raw_name(), val_var);
    } else if (dt == DataType::u32) {
      emit(
          "const auto {} = atomic_fetch_{}_explicit((device atomic_uint*){}, "
          "{}, "
          "metal::memory_order_relaxed);",
          stmt->raw_name(), op_name, stmt->dest->raw_name(), val_var);
    } else if (dt == DataType::f32) {
      if (handle_float) {
        emit("const float {} = fatomic_fetch_{}({}, {});", stmt->raw_name(),
             op_name, stmt->dest->raw_name(), val_var);
      } else {
        TI_ERROR("Metal does not support atomic {} for floating points",
                 op_name);
      }
    } else {
      TI_ERROR("Metal only supports 32-bit atomic data types");
    }

    if (use_simd_in_tls_reduction) {
      current_appender().pop_indent();
      emit("}}");  // closes `if (kKernelTidInSimdgroupName == 0) {`
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
    auto loop_var_name = for_stmt->raw_name();
    if (!for_stmt->reversed) {
      emit("for (int {}_ = {}; {}_ < {}; {}_ = {}_ + {}) {{", loop_var_name,
           for_stmt->begin->raw_name(), loop_var_name,
           for_stmt->end->raw_name(), loop_var_name, loop_var_name, 1);
      emit("  int {} = {}_;", loop_var_name, loop_var_name);
    } else {
      // reversed for loop
      emit("for (int {}_ = {} - 1; {}_ >= {}; {}_ = {}_ - {}) {{",
           loop_var_name, for_stmt->end->raw_name(), loop_var_name,
           for_stmt->begin->raw_name(), loop_var_name, loop_var_name, 1);
      emit("  int {} = {}_;", loop_var_name, loop_var_name);
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
    } else if (stmt->task_type == Type::struct_for) {
      generate_struct_for_kernel(stmt);
    } else if (stmt->task_type == Type::clear_list) {
      add_runtime_list_op_kernel(stmt, "clear_list");
    } else if (stmt->task_type == Type::listgen) {
      add_runtime_list_op_kernel(stmt, "element_listgen");
    } else if (stmt->task_type == Type::gc) {
      // Ignored
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

  void visit(ContinueStmt *stmt) override {
    if (stmt->as_return()) {
      emit("return;");
    } else {
      emit("continue;");
    }
  }

  void visit(WhileStmt *stmt) override {
    emit("while (true) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(RandStmt *stmt) override {
    emit("const auto {} = metal_rand_{}({});", stmt->raw_name(),
         data_type_short_name(stmt->ret_type.data_type), kRandStateVarName);
  }

  void visit(PrintStmt *stmt) override {
    used_features()->print = true;

    const auto &contents = stmt->contents;
    const int num_entries = contents.size();
    const std::string msgbuf_var_name = stmt->raw_name() + "_msgbuf_";
    emit("device auto* {} = mtl_print_alloc_buf({}, {});", msgbuf_var_name,
         kPrintAllocVarName, num_entries);
    // Check for buffer overflow
    emit("if ({}) {{", msgbuf_var_name);
    {
      ScopedIndent s(current_appender());
      const std::string msg_var_name = stmt->raw_name() + "_msg_";
      emit("PrintMsg {}({}, {});", msg_var_name, msgbuf_var_name, num_entries);
      for (int i = 0; i < num_entries; ++i) {
        const auto &entry = contents[i];
        if (std::holds_alternative<Stmt *>(entry)) {
          auto *arg_stmt = std::get<Stmt *>(entry);
          const auto dt = arg_stmt->element_type();
          TI_ASSERT_INFO(dt == DataType::i32 || dt == DataType::f32,
                         "print() only supports i32 or f32 scalars for now.");
          emit("{}.pm_set_{}({}, {});", msg_var_name, data_type_short_name(dt),
               i, arg_stmt->raw_name());
        } else {
          const int str_id = print_strtab_->put(std::get<std::string>(entry));
          emit("{}.pm_set_str({}, {});", msg_var_name, i, str_id);
        }
      }
    }
    emit("}}");
  }

  void visit(StackAllocaStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);

    const auto &var_name = stmt->raw_name();
    emit("byte {}[{}];", var_name, stmt->size_in_bytes());
    emit("mtl_ad_stack_init({});", var_name);
  }

  void visit(StackPopStmt *stmt) override {
    emit("mtl_ad_stack_pop({});", stmt->stack->raw_name());
  }

  void visit(StackPushStmt *stmt) override {
    auto *stack = stmt->stack->as<StackAllocaStmt>();
    const auto &stack_name = stack->raw_name();
    const auto elem_size = stack->element_size_in_bytes();
    emit("mtl_ad_stack_push({}, {});", stack_name, elem_size);
    const auto primal_name = stmt->raw_name() + "_primal_";
    emit(
        "thread auto* {} = reinterpret_cast<thread "
        "{}*>(mtl_ad_stack_top_primal({}, {}));",
        primal_name, metal_data_type_name(stmt->element_type()), stack_name,
        elem_size);
    emit("*{} = {};", primal_name, stmt->v->raw_name());
  }

  void visit(StackLoadTopStmt *stmt) override {
    auto *stack = stmt->stack->as<StackAllocaStmt>();
    const auto primal_name = stmt->raw_name() + "_primal_";
    emit(
        "thread auto* {} = reinterpret_cast<thread "
        "{}*>(mtl_ad_stack_top_primal({}, {}));",
        primal_name, metal_data_type_name(stmt->element_type()),
        stack->raw_name(), stack->element_size_in_bytes());
    emit("const auto {} = *{};", stmt->raw_name(), primal_name);
  }

  void visit(StackLoadTopAdjStmt *stmt) override {
    auto *stack = stmt->stack->as<StackAllocaStmt>();
    const auto adjoint_name = stmt->raw_name() + "_adjoint_";
    emit(
        "thread auto* {} = reinterpret_cast<thread "
        "{}*>(mtl_ad_stack_top_adjoint({}, {}));",
        adjoint_name, metal_data_type_name(stmt->element_type()),
        stack->raw_name(), stack->element_size_in_bytes());
    emit("const auto {} = *{};", stmt->raw_name(), adjoint_name);
  }

  void visit(StackAccAdjointStmt *stmt) override {
    auto *stack = stmt->stack->as<StackAllocaStmt>();
    const auto adjoint_name = stmt->raw_name() + "_adjoint_";
    emit(
        "thread auto* {} = reinterpret_cast<thread "
        "{}*>(mtl_ad_stack_top_adjoint({}, {}));",
        adjoint_name, metal_data_type_name(stmt->element_type()),
        stack->raw_name(), stack->element_size_in_bytes());
    emit("*{} += {};", adjoint_name, stmt->v->raw_name());
  }

 private:
  void emit_headers() {
    SectionGuard sg(this, Section::Headers);
    emit("#include <metal_stdlib>");
    emit("#include <metal_compute>");
    emit("using namespace metal;");
  }

  void generate_structs() {
    SectionGuard sg(this, Section::Structs);
    emit("using byte = char;");
    emit("");
    current_appender().append_raw(shaders::kMetalHelpersSourceCode);
    emit("");
    current_appender().append_raw(compiled_structs_->runtime_utils_source_code);
    emit("");
    current_appender().append_raw(compiled_structs_->snode_structs_source_code);
    emit("");
    current_appender().append_raw(shaders::kMetalAdStackSourceCode);
    emit("");
    current_appender().append_raw(shaders::kMetalPrintSourceCode);
    emit("");
    emit_kernel_args_struct();
  }

  void emit_kernel_args_struct() {
    if (ctx_attribs_.empty()) {
      return;
    }
    const auto class_name = kernel_args_classname();
    emit("class {} {{", class_name);
    emit(" public:");
    {
      ScopedIndent s(current_appender());
      emit("explicit {}(device byte* addr) : addr_(addr) {{}}", class_name);
      for (const auto &arg : ctx_attribs_.args()) {
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
      for (const auto &ret : ctx_attribs_.rets()) {
        const auto dt_name = metal_data_type_name(ret.dt);
        emit("device {}* ret{}() {{", dt_name, ret.index);
        if (ret.is_array) {
          emit("  // array, size={} B", ret.stride);
        } else {
          emit("  // scalar, size={} B", ret.stride);
        }
        emit("  return (device {}*)(addr_ + {});", dt_name, ret.offset_in_mem);
        emit("}}");
      }
      emit("");
      emit("int32_t extra_arg(int i, int j) {{");
      emit("  device int32_t* base = (device int32_t*)(addr_ + {});",
           ctx_attribs_.ctx_bytes());
      emit("  return *(base + (i * {}) + j);", taichi_max_num_indices);
      emit("}}");
    }
    emit(" private:");
    emit("  device byte* addr_;");
    emit("}};");
  }

  void generate_kernels() {
    SectionGuard sg(this, Section::Kernels);
    kernel_->ir->accept(this);

    if (used_features()->sparse) {
      emit("");
      current_appender().append_raw(shaders::kMetalRuntimeKernelsSourceCode);
    }
  }

  std::vector<BuffersEnum> get_common_buffers() {
    std::vector<BuffersEnum> result;
    if (needs_root_buffer_) {
      result.push_back(BuffersEnum::Root);
    }
    result.push_back(BuffersEnum::GlobalTmps);
    if (!ctx_attribs_.empty()) {
      result.push_back(BuffersEnum::Context);
    }
    result.push_back(BuffersEnum::Runtime);
    // TODO(k-ye): Bind this buffer only when print() is used.
    result.push_back(BuffersEnum::Print);
    return result;
  }

  void generate_serial_kernel(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::serial);
    const std::string mtl_kernel_name = make_kernel_name();
    KernelAttributes ka;
    ka.name = mtl_kernel_name;
    ka.task_type = stmt->task_type;
    ka.buffers = get_common_buffers();
    ka.num_threads = 1;

    emit_mtl_kernel_sig(mtl_kernel_name, ka.buffers);
    {
      ScopedIndent s(current_appender());
      emit("// serial");
      emit("if ({} > 0) return;", kKernelThreadIdName);

      current_kernel_attribs_ = &ka;
      const auto mtl_func_name = mtl_kernel_func_name(mtl_kernel_name);
      emit_mtl_kernel_func_def(mtl_func_name, ka.buffers, stmt->body.get());
      emit_call_mtl_kernel_func(mtl_func_name, ka.buffers,
                                /*loop_index_expr=*/"0");
    }
    // Close kernel
    emit("}}\n");
    current_kernel_attribs_ = nullptr;

    mtl_kernels_attribs()->push_back(ka);
  }

  void generate_range_for_kernel(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::range_for);
    const std::string mtl_kernel_name = make_kernel_name();
    KernelAttributes ka;
    ka.name = mtl_kernel_name;
    ka.task_type = stmt->task_type;
    ka.buffers = get_common_buffers();

    const bool used_tls = (stmt->tls_prologue != nullptr);
    KernelSigExtensions kernel_exts;
    kernel_exts.use_simdgroup = (used_tls && cgen_config_.allow_simdgroup);
    used_features()->simdgroup =
        used_features()->simdgroup || kernel_exts.use_simdgroup;

    emit_mtl_kernel_sig(mtl_kernel_name, ka.buffers, kernel_exts);

    ka.range_for_attribs = KernelAttributes::RangeForAttributes();
    auto &range_for_attribs = ka.range_for_attribs.value();
    range_for_attribs.const_begin = stmt->const_begin;
    range_for_attribs.const_end = stmt->const_end;
    range_for_attribs.begin =
        (stmt->const_begin ? stmt->begin_value : stmt->begin_offset);
    range_for_attribs.end =
        (stmt->const_end ? stmt->end_value : stmt->end_offset);

    current_appender().push_indent();
    const std::string total_elems_name("total_elems");
    // Begin expression of the for range, this can be either a constant
    // (const_begin == true), or a variable loaded from the global temporaries.
    std::string begin_expr;
    if (range_for_attribs.const_range()) {
      const int num_elems = range_for_attribs.end - range_for_attribs.begin;
      begin_expr = std::to_string(stmt->begin_value);
      emit("// range_for, range known at compile time");
      emit("const int {} = {};", total_elems_name, num_elems);
      // We don't clamp this to kMaxNumThreadsGridStrideLoop, because we know
      // for sure that we need |num_elems| of threads.
      // sdf_renderer.py benchmark for setting |num_threads|
      // - num_elemnts: ~20 samples/s
      // - kMaxNumThreadsGridStrideLoop: ~12 samples/s
      ka.num_threads = num_elems;
    } else {
      emit("// range_for, range known at runtime");
      begin_expr = stmt->const_begin
                       ? std::to_string(stmt->begin_value)
                       : inject_load_global_tmp(stmt->begin_offset);
      const auto end_expr = stmt->const_end
                                ? std::to_string(stmt->end_value)
                                : inject_load_global_tmp(stmt->end_offset);
      emit("const int {} = {} - {};", total_elems_name, end_expr, begin_expr);
      ka.num_threads = kMaxNumThreadsGridStrideLoop;
    }
    // begin_ = thread_id   + begin_expr
    emit("const int begin_ = {} + {};", kKernelThreadIdName, begin_expr);
    // end_   = total_elems + begin_expr
    emit("const int end_ = {} + {};", total_elems_name, begin_expr);

    if (used_tls) {
      // Using TLS means we will access some SNodes within this kernel. The
      // struct of an SNode needs Runtime and MemoryAllocator to construct.
      emit_runtime_and_memalloc_def();
      // Using |int32_t| because it aligns to 4bytes.
      emit("// TLS prologue");
      const std::string tls_bufi32_name = "tls_bufi32_";
      emit("int32_t {}[{}];", tls_bufi32_name, (stmt->tls_size + 3) / 4);
      emit("thread char* {} = reinterpret_cast<thread char*>({});",
           kTlsBufferName, tls_bufi32_name);
      stmt->tls_prologue->accept(this);
    }

    emit("for (int ii = begin_; ii < end_; ii += {}) {{", kKernelGridSizeName);
    {
      ScopedIndent s2(current_appender());

      current_kernel_attribs_ = &ka;
      const auto mtl_func_name = mtl_kernel_func_name(mtl_kernel_name);
      std::vector<FuncParamLiteral> extra_func_params;
      std::vector<std::string> extra_args;
      if (used_tls) {
        extra_func_params.push_back({"thread char*", kTlsBufferName});
        extra_args.push_back(kTlsBufferName);
      }
      emit_mtl_kernel_func_def(mtl_func_name, ka.buffers, extra_func_params,
                               stmt->body.get());
      emit_call_mtl_kernel_func(mtl_func_name, ka.buffers, extra_args,
                                /*loop_index_expr=*/"ii");
    }
    emit("}}");  // closes for loop

    if (used_tls) {
      TI_ASSERT(stmt->tls_epilogue != nullptr);
      inside_tls_epilogue_ = true;
      emit("{{  // TLS epilogue");
      stmt->tls_epilogue->accept(this);
      inside_tls_epilogue_ = false;
      emit("}}");
    }

    current_appender().pop_indent();
    // Close kernel
    emit("}}\n");
    current_kernel_attribs_ = nullptr;

    mtl_kernels_attribs()->push_back(ka);
  }

  void generate_struct_for_kernel(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::struct_for);
    const std::string mtl_kernel_name = make_kernel_name();

    KernelAttributes ka;
    ka.name = mtl_kernel_name;
    ka.task_type = stmt->task_type;
    ka.buffers = get_common_buffers();

    emit_mtl_kernel_sig(mtl_kernel_name, ka.buffers);

    const int sn_id = stmt->snode->id;
    // struct_for kernels use grid-stride loops
    const int total_num_elems_from_root =
        compiled_structs_->snode_descriptors.find(sn_id)
            ->second.total_num_elems_from_root;
    ka.num_threads =
        std::min(total_num_elems_from_root, kMaxNumThreadsGridStrideLoop);

    current_appender().push_indent();
    emit("// struct_for");
    emit_runtime_and_memalloc_def();
    emit("ListManager parent_list;");
    emit("parent_list.lm_data = ({}->snode_lists + {});", kRuntimeVarName,
         sn_id);
    emit("parent_list.mem_alloc = {};", kMemAllocVarName);
    emit("const SNodeMeta parent_meta = {}->snode_metas[{}];", kRuntimeVarName,
         sn_id);
    emit("const int child_stride = parent_meta.element_stride;");
    emit("const int child_num_slots = parent_meta.num_slots;");
    // Grid-stride loops:
    // Each thread begins at thread_index, and incremets by grid_size
    emit("for (int ii = {}; ii < {}; ii += {}) {{", kKernelThreadIdName,
         total_num_elems_from_root, kKernelGridSizeName);
    {
      ScopedIndent s2(current_appender());
      emit("const int parent_idx_ = (ii / child_num_slots);");
      emit("if (parent_idx_ >= parent_list.num_active()) return;");
      emit("const int child_idx_ = (ii % child_num_slots);");
      emit(
          "const auto parent_elem_ = "
          "parent_list.get<ListgenElement>(parent_idx_);");
      emit("device auto *parent_addr_ = {} + parent_elem_.root_mem_offset;",
           kRootBufferName);
      emit("if (!is_active(parent_addr_, parent_meta, child_idx_)) continue;");
      emit("ListgenElement {};", kListgenElemVarName);
      // No need to add mem_offset_in_parent, because place() always starts at 0
      emit(
          "{}.root_mem_offset = parent_elem_.root_mem_offset + child_idx_ * "
          "child_stride;",
          kListgenElemVarName);
      emit(
          "refine_coordinates(parent_elem_, {}->snode_extractors[{}], "
          "child_idx_, &{});",
          kRuntimeVarName, sn_id, kListgenElemVarName);

      current_kernel_attribs_ = &ka;
      const auto mtl_func_name = mtl_kernel_func_name(mtl_kernel_name);
      emit_mtl_kernel_func_def(
          mtl_func_name, ka.buffers,
          /*extra_params=*/
          {{"thread const ListgenElement&", kListgenElemVarName}},
          stmt->body.get());
      emit_call_mtl_kernel_func(mtl_func_name, ka.buffers,
                                /*extra_args=*/
                                {kListgenElemVarName},
                                /*loop_index_expr=*/"ii");
      current_kernel_attribs_ = nullptr;
    }
    emit("}}");  // closes for loop
    current_appender().pop_indent();
    emit("}}\n");  // closes kernel

    mtl_kernels_attribs()->push_back(ka);
  }

  void add_runtime_list_op_kernel(OffloadedStmt *stmt,
                                  const std::string &kernel_name) {
    using Type = OffloadedStmt::TaskType;
    const auto type = stmt->task_type;
    auto *const sn = stmt->snode;
    KernelAttributes ka;
    ka.name = kernel_name;
    ka.task_type = stmt->task_type;
    if (type == Type::clear_list) {
      ka.num_threads = 1;
      ka.buffers = {BuffersEnum::Runtime, BuffersEnum::Context};
    } else if (type == Type::listgen) {
      // listgen kernels use grid-stride loops
      const auto &sn_descs = compiled_structs_->snode_descriptors;
      ka.num_threads = std::min(
          sn_descs.find(sn->id)->second.total_num_self_from_root(sn_descs),
          kMaxNumThreadsGridStrideLoop);
      ka.buffers = {BuffersEnum::Runtime, BuffersEnum::Root,
                    BuffersEnum::Context};
    } else {
      TI_ERROR("Unsupported offload task type {}", stmt->task_name());
    }

    ka.runtime_list_op_attribs = KernelAttributes::RuntimeListOpAttributes();
    ka.runtime_list_op_attribs->snode = sn;
    current_kernel_attribs_ = nullptr;

    mtl_kernels_attribs()->push_back(ka);
    used_features()->sparse = true;
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

  struct FuncParamLiteral {
    std::string type;
    std::string name;
  };

  void emit_mtl_kernel_func_def(
      const std::string &kernel_func_name,
      const std::vector<KernelAttributes::Buffers> &buffers,
      const std::vector<FuncParamLiteral> &extra_params,
      Block *func_ir) {
    SectionGuard sg(this, Section::KernelFuncs);

    emit("void {}(", kernel_func_name);
    for (auto b : buffers) {
      emit("    device byte* {},", buffer_to_name(b));
    }
    for (const auto &p : extra_params) {
      emit("    {} {},", p.type, p.name);
    }
    emit("    const int {}) {{", kLinearLoopIndexName);

    {
      ScopedIndent s(current_appender());
      emit_runtime_and_memalloc_def();
      if (!ctx_attribs_.empty()) {
        emit("{} {}({});", kernel_args_classname(), kContextVarName,
             kContextBufferName);
      }
      // Init RandState
      emit(
          "device {rty}* {rand} = reinterpret_cast<device "
          "{rty}*>({rtm}->rand_seeds + ({lidx} % {nums}));",
          fmt::arg("rty", "RandState"), fmt::arg("rand", kRandStateVarName),
          fmt::arg("rtm", kRuntimeVarName),
          fmt::arg("lidx", kLinearLoopIndexName),
          fmt::arg("nums", kNumRandSeeds));
      // Init PrintMsgAllocator
      emit("device auto* {} = reinterpret_cast<device PrintMsgAllocator*>({});",
           kPrintAllocVarName, kPrintBufferName);
    }
    // We do not need additional indentation, because |func_ir| itself is a
    // block, which will be indented automatically.
    func_ir->accept(this);

    emit("}}\n");
  }

  inline void emit_mtl_kernel_func_def(
      const std::string &kernel_func_name,
      const std::vector<KernelAttributes::Buffers> &buffers,
      Block *func_ir) {
    emit_mtl_kernel_func_def(kernel_func_name, buffers, /*extra_params=*/{},
                             func_ir);
  }

  void emit_call_mtl_kernel_func(
      const std::string &kernel_func_name,
      const std::vector<KernelAttributes::Buffers> &buffers,
      const std::vector<std::string> &extra_args,
      const std::string &loop_index_expr) {
    TI_ASSERT(code_section_ == Section::Kernels);
    std::string call = kernel_func_name + "(";
    for (auto b : buffers) {
      call += buffer_to_name(b) + ", ";
    }
    for (const auto &a : extra_args) {
      call += a + ", ";
    }
    call += fmt::format("{});", loop_index_expr);
    emit(std::move(call));
  }

  inline void emit_call_mtl_kernel_func(
      const std::string &kernel_func_name,
      const std::vector<KernelAttributes::Buffers> &buffers,
      const std::string &loop_index_expr) {
    emit_call_mtl_kernel_func(kernel_func_name, buffers, /*extra_args=*/{},
                              loop_index_expr);
  }

  struct KernelSigExtensions {
    // https://stackoverflow.com/a/44693603/12003165
    KernelSigExtensions() noexcept {
    }

    bool use_simdgroup = false;
  };

  void emit_mtl_kernel_sig(
      const std::string &kernel_name,
      const std::vector<KernelAttributes::Buffers> &buffers,
      const KernelSigExtensions &exts = {}) {
    emit("kernel void {}(", kernel_name);
    for (int i = 0; i < buffers.size(); ++i) {
      emit("    device byte* {} [[buffer({})]],", buffer_to_name(buffers[i]),
           i);
    }
    emit("    const uint {} [[threads_per_grid]],", kKernelGridSizeName);
    if (exts.use_simdgroup) {
      emit("    const uint {} [[thread_index_in_simdgroup]],",
           kKernelTidInSimdgroupName);
    }
    emit("    const uint {} [[thread_position_in_grid]]) {{",
         kKernelThreadIdName);
  }

  void emit_runtime_and_memalloc_def() {
    emit("device auto *{} = reinterpret_cast<device Runtime *>({});",
         kRuntimeVarName, kRuntimeBufferName);
    emit(
        "device auto *{} = reinterpret_cast<device MemoryAllocator *>({} + 1);",
        kMemAllocVarName, kRuntimeVarName);
  }

  std::string make_kernel_name() {
    return fmt::format("{}_{}", mtl_kernel_prefix_, mtl_kernel_count_++);
  }

  inline std::string kernel_args_classname() const {
    return fmt::format("{}_args", mtl_kernel_prefix_);
  }

  static inline std::string mtl_kernel_func_name(
      const std::string &kernel_name) {
    return kernel_name + "_func";
  }

  class SectionGuard {
   public:
    SectionGuard(KernelCodegen *kg, Section new_sec)
        : kg_(kg), saved_(kg->code_section_) {
      kg->code_section_ = new_sec;
    }

    ~SectionGuard() {
      kg_->code_section_ = saved_;
    }

   private:
    KernelCodegen *const kg_;
    const Section saved_;
  };

  friend class SectionGuard;

  const LineAppender &current_appender() const {
    return section_appenders_.find(code_section_)->second;
  }
  LineAppender &current_appender() {
    return section_appenders_[code_section_];
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    current_appender().append(std::move(f), std::forward<Args>(args)...);
  }

  std::vector<KernelAttributes> *mtl_kernels_attribs() {
    return &(ti_kernel_attribus_.mtl_kernels_attribs);
  }

  TaichiKernelAttributes::UsedFeatures *used_features() {
    return &(ti_kernel_attribus_.used_features);
  }

  const std::string mtl_kernel_prefix_;
  const std::string root_snode_type_name_;
  Kernel *const kernel_;
  const CompiledStructs *const compiled_structs_;
  const bool needs_root_buffer_;
  const KernelContextAttributes ctx_attribs_;
  PrintStringTable *const print_strtab_;
  const CodeGen::Config &cgen_config_;

  bool is_top_level_{true};
  int mtl_kernel_count_{0};
  TaichiKernelAttributes ti_kernel_attribus_;
  GetRootStmt *root_stmt_{nullptr};
  KernelAttributes *current_kernel_attribs_{nullptr};
  bool inside_tls_epilogue_{false};
  Section code_section_{Section::Structs};
  std::unordered_map<Section, LineAppender> section_appenders_;
};

}  // namespace

CodeGen::CodeGen(Kernel *kernel,
                 KernelManager *kernel_mgr,
                 const CompiledStructs *compiled_structs,
                 const Config &config)
    : kernel_(kernel),
      kernel_mgr_(kernel_mgr),
      compiled_structs_(compiled_structs),
      id_(Program::get_kernel_id()),
      taichi_kernel_name_(fmt::format("mtl_k{:04d}_{}", id_, kernel_->name)),
      config_(config) {
}

FunctionType CodeGen::compile() {
  auto &config = kernel_->program.config;
  config.demote_dense_struct_fors = true;
  irpass::compile_to_executable(kernel_->ir.get(), config,
                                /*vectorize=*/false, kernel_->grad,
                                /*ad_use_stack=*/true, config.print_ir,
                                /*lower_global_access=*/true,
                                /*make_thread_local=*/config.make_thread_local);

  KernelCodegen codegen(
      taichi_kernel_name_, kernel_->program.snode_root->node_type_name, kernel_,
      compiled_structs_, kernel_mgr_->print_strtable(), config_);
  const auto source_code = codegen.run();
  kernel_mgr_->register_taichi_kernel(taichi_kernel_name_, source_code,
                                      codegen.ti_kernels_attribs(),
                                      codegen.kernel_ctx_attribs());
  return [kernel_mgr = kernel_mgr_,
          kernel_name = taichi_kernel_name_](Context &ctx) {
    kernel_mgr->launch_taichi_kernel(kernel_name, &ctx);
  };
}

}  // namespace metal
TLANG_NAMESPACE_END
