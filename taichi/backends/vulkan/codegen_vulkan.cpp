#include "taichi/backends/vulkan/codegen_vulkan.h"

#include <string>
#include <vector>

#include "taichi/program/program.h"
#include "taichi/program/kernel.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/ir.h"
#include "taichi/util/line_appender.h"
#include "taichi/backends/vulkan/kernel_utils.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/opengl/opengl_data_types.h"
#include "taichi/ir/transforms.h"

namespace taichi {
namespace lang {
namespace vulkan {
namespace {

constexpr char kRootBufferName[] = "root_buffer";
constexpr char kGlobalTmpsBufferName[] = "global_tmps_buffer";
constexpr char kContextBufferName[] = "context_buffer";

constexpr char kGlobalInvocationIDName[] = "int(gl_GlobalInvocationID.x)";
constexpr char kLinearLoopIndexName[] = "linear_loop_idx_";

constexpr int kMaxNumThreadsGridStrideLoop = 65536;

#define TI_INSIDE_VULKAN_CODEGEN
#include "taichi/backends/vulkan/shaders/atomics.glsl.h"
#undef TI_INSIDE_VULKAN_CODEGEN

using opengl::opengl_data_type_name;
using BuffersEnum = TaskAttributes::Buffers;
using BufferBind = TaskAttributes::BufferBind;

std::string buffer_instance_name(BuffersEnum b) {
  // https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Syntax
  switch (b) {
    case BuffersEnum::Root:
      return kRootBufferName;
    case BuffersEnum::GlobalTmps:
      return kGlobalTmpsBufferName;
    case BuffersEnum::Context:
      return kContextBufferName;
    default:
      TI_NOT_IMPLEMENTED;
      break;
  }
  return {};
}

std::string store_as_int_bits(const std::string &in, DataType dt) {
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return fmt::format("floatBitsToInt({})", in);
  }
  return in;
}

std::string load_from_int_bits(const std::string &in, DataType dt) {
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return fmt::format("intBitsToFloat({})", in);
  }
  return in;
}

std::string vk_data_address_shifter(const Stmt *s, DataType) {
  // Hardcoded ">> 2" because we only support 32-bit for now.
  return fmt::format("({} >> 2)", s->raw_name());
}

class TaskCodegen : public IRVisitor {
 private:
  enum class Section {
    Headers,
    Kernels,
  };

  static constexpr Section kAllSections[] = {
      Section::Headers,
      Section::Kernels,
  };

 public:
  struct Params {
    OffloadedStmt *task_ir;
    const CompiledSNodeStructs *compiled_structs;
    const KernelContextAttributes *ctx_attribs;
    std::string ti_kernel_name;
    int task_id_in_kernel;
  };

  explicit TaskCodegen(const Params &params)
      : task_ir_(params.task_ir),
        compiled_structs_(params.compiled_structs),
        ctx_attribs_(params.ctx_attribs),
        task_name_(fmt::format("{}_t{:02d}",
                               params.ti_kernel_name,
                               params.task_id_in_kernel)) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  struct Result {
    std::string source_code;
    TaskAttributes task_attribs;
  };

  Result run() {
    code_section_ = Section::Kernels;
    if (task_ir_->task_type == OffloadedTaskType::serial) {
      generate_serial_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::range_for) {
      // struct_for is automatically lowered to ranged_for for dense snodes
      generate_range_for_kernel(task_ir_);
    } else {
      TI_ERROR("Unsupported offload type={} on Vulkan arch",
               task_ir_->task_name());
    }
    // Headers need global information, so it has to be delayed after visiting
    // the task IR.
    emit_headers();

    Result res;
    for (const auto s : kAllSections) {
      res.source_code += section_appenders_.find(s)->second.lines();
      res.source_code += '\n';
    }
    res.task_attribs = std::move(task_attribs_);
    return res;
  }

  void visit(OffloadedStmt *) override {
    TI_ERROR("This codegen is supposed to deal with one offloaded task");
  }

  void visit(Block *stmt) override {
    push_indent();
    for (auto &s : stmt->statements) {
      s->accept(this);
    }
    pop_indent();
  }

  void visit(ConstStmt *const_stmt) override {
    TI_ASSERT(const_stmt->width() == 1);
    emit("const {} {} = {};", opengl_data_type_name(const_stmt->element_type()),
         const_stmt->raw_name(), const_stmt->val[0].stringify());
  }

  void visit(AllocaStmt *alloca) override {
    emit("{} {} = 0;", opengl_data_type_name(alloca->element_type()),
         alloca->raw_name());
  }

  void visit(LocalLoadStmt *stmt) override {
    // TODO: optimize for partially vectorized load...
    bool linear_index = true;
    for (int i = 0; i < (int)stmt->src.size(); i++) {
      if (stmt->src[i].offset != i) {
        linear_index = false;
      }
    }
    if (stmt->same_source() && linear_index &&
        stmt->width() == stmt->src[0].var->width()) {
      auto ptr = stmt->src[0].var;
      emit("const {} {} = {};", opengl_data_type_name(stmt->element_type()),
           stmt->raw_name(), ptr->raw_name());
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    emit("{} = {};", stmt->dest->raw_name(), stmt->val->raw_name());
  }

  void visit(GetRootStmt *stmt) override {
    // Should we assert |root_stmt_| is assigned only once?
    root_stmt_ = stmt;
    emit("const int {} = 0;", stmt->raw_name());
  }

  void visit(GetChStmt *stmt) override {
    // TODO: GetChStmt -> GetComponentStmt ?
    const auto &snode_descs = compiled_structs_->snode_descriptors;
    auto *out_snode = stmt->output_snode;
    TI_ASSERT(snode_descs.at(stmt->input_snode->id).get_child(stmt->chid) ==
              out_snode);

    emit("// SNode: {} -> {}", stmt->input_snode->node_type_name,
         out_snode->node_type_name);
    emit("const int {} = {} + {};", stmt->raw_name(),
         stmt->input_ptr->raw_name(),
         snode_descs.at(out_snode->id).mem_offset_in_parent_cell);
    if (out_snode->is_place()) {
      TI_ASSERT(ptr_to_buffers_.count(stmt) == 0);
      ptr_to_buffers_[stmt] = BuffersEnum::Root;
    }
  }

  void visit(SNodeLookupStmt *stmt) override {
    // TODO: SNodeLookupStmt -> GetSNodeCellStmt ?
    std::string parent;
    if (stmt->input_snode) {
      parent = stmt->input_snode->raw_name();
    } else {
      TI_ASSERT(root_stmt_ != nullptr);
      parent = root_stmt_->raw_name();
    }
    const auto *sn = stmt->snode;

    if (stmt->activate && !(sn->type == SNodeType::dense)) {
      // Sparse SNode not supported yet.
      TI_NOT_IMPLEMENTED;
    }
    const auto &snode_descs = compiled_structs_->snode_descriptors;
    emit("// Get the cell of SNode {}", sn->node_type_name);
    emit("const int {} = {} + ({} * {});", stmt->raw_name(), parent,
         stmt->input_index->raw_name(), snode_descs.at(sn->id).cell_stride);
  }

  void visit(LinearizeStmt *stmt) override {
    std::string val = "0";
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      val = fmt::format("({} * {} + {})", val, stmt->strides[i],
                        stmt->inputs[i]->raw_name());
    }
    emit("const int {} = {};", stmt->raw_name(), val);
  }

  void visit(BitExtractStmt *stmt) override {
    emit("const int {} = (({} >> {}) & ((1 << {}) - 1));", stmt->raw_name(),
         stmt->input->raw_name(), stmt->bit_begin,
         stmt->bit_end - stmt->bit_begin);
  }

  void visit(LoopIndexStmt *stmt) override {
    const auto stmt_name = stmt->raw_name();
    if (stmt->loop->is<OffloadedStmt>()) {
      const auto type = stmt->loop->as<OffloadedStmt>()->task_type;
      if (type == OffloadedTaskType::range_for) {
        TI_ASSERT(stmt->index == 0);
        emit("const int {} = {};", stmt_name, kLinearLoopIndexName);
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

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    const auto dt = stmt->val->element_type();
    emit("{} = {};", at_buffer(stmt->dest, dt),
         store_as_int_bits(stmt->val->raw_name(), dt));
  }

  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt = stmt->element_type();
    const auto loaded_int = at_buffer(stmt->src, dt);
    emit("const {} {} = {};", opengl_data_type_name(dt), stmt->raw_name(),
         load_from_int_bits(loaded_int, dt));
  }

  void visit(ArgLoadStmt *stmt) override {
    const auto arg_id = stmt->arg_id;
    const auto &arg_attribs = ctx_attribs_->args()[arg_id];
    const auto offset_in_mem = arg_attribs.offset_in_mem;
    if (stmt->is_ptr) {
      emit("// Pointer arg: id={} offset_in_mem={}", arg_id, offset_in_mem);
      // Do not shift! We are indexing the buffers at byte granularity.
      emit("const int {} = {};", stmt->raw_name(), offset_in_mem);
    } else {
      const auto dt = arg_attribs.dt;
      const auto val_str = fmt::format("{}[{}]", kContextBufferName,
                                       (offset_in_mem / sizeof(int32_t)));
      emit("// Scalar arg: id={} offset_in_mem={}", arg_id, offset_in_mem);
      emit("const {} {} = {};", opengl_data_type_name(dt), stmt->raw_name(),
           load_from_int_bits(val_str, dt));
    }
  }

  void visit(ReturnStmt *stmt) override {
    // TODO: use stmt->ret_id instead of 0 as index
    const auto &ret_attribs = ctx_attribs_->rets()[0];
    const int index_in_buffer = ret_attribs.offset_in_mem / sizeof(int32_t);
    emit("// Return value: offset_in_mem={}", ret_attribs.offset_in_mem);
    emit("{}[{}] = {};", kContextBufferName, index_in_buffer,
         store_as_int_bits(stmt->value->raw_name(), ret_attribs.dt));
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    const auto dt = opengl_data_type_name(stmt->element_type().ptr_removed());
    emit("const int {} = {}", stmt->raw_name(), stmt->offset);
    ptr_to_buffers_[stmt] = BuffersEnum::GlobalTmps;
  }

  void visit(ExternalPtrStmt *stmt) override {
    // Used mostly for transferring data between host (e.g. numpy array) and
    // Vulkan.
    TI_ASSERT(stmt->width() == 1);
    const auto linear_offset_name =
        fmt::format("{}_linear_mem_offset_", stmt->raw_name());
    emit("int {} = 0;", linear_offset_name);
    emit("{{");
    {
      ScopedIndent s(current_appender());
      const auto *argload = stmt->base_ptrs[0]->as<ArgLoadStmt>();
      const int arg_id = argload->arg_id;
      const int num_indices = stmt->indices.size();
      std::vector<std::string> size_var_names;
      const auto extra_args_mem_offset = ctx_attribs_->extra_args_mem_offset();
      const auto extra_args_index_base =
          (extra_args_mem_offset / sizeof(int32_t));
      emit("// External ptr, extra args: mem_offset={} index_base={}",
           extra_args_mem_offset, extra_args_index_base);
      for (int i = 0; i < num_indices; i++) {
        std::string var_name = fmt::format("{}_size{}_", stmt->raw_name(), i);
        const auto extra_arg_linear_index_offset =
            (arg_id * taichi_max_num_indices) + i;
        const auto extra_arg_linear_index =
            extra_args_index_base + extra_arg_linear_index_offset;
        emit("// Extra arg: arg_id={} i={} linear_index=({} + {})={}", arg_id,
             i, extra_args_index_base, extra_arg_linear_index_offset,
             extra_arg_linear_index);
        emit("const int {} = {}[{}];", var_name, kContextBufferName,
             extra_arg_linear_index);
        size_var_names.push_back(std::move(var_name));
      }
      for (int i = 0; i < num_indices; i++) {
        emit("{} *= {};", linear_offset_name, size_var_names[i]);
        emit("{} += {};", linear_offset_name, stmt->indices[i]->raw_name());
      }
      emit("// Convert index to bytes");
      emit("{} = ({} << 2);", linear_offset_name, linear_offset_name);
    }
    emit("}}");
    emit("const int {} = ({} + {});", stmt->raw_name(),
         stmt->base_ptrs[0]->raw_name(), linear_offset_name);
    ptr_to_buffers_[stmt] = BuffersEnum::Context;
  }

  void visit(UnaryOpStmt *stmt) override {
    const auto dt_name = opengl_data_type_name(stmt->element_type());
    const auto var_decl = fmt::format("const {} {}", dt_name, stmt->raw_name());
    const auto operand_name = stmt->operand->raw_name();

    if (stmt->op_type == UnaryOpType::logic_not) {
      emit("{} = {}({} == 0);", var_decl, dt_name, operand_name);
    } else if (stmt->op_type == UnaryOpType::neg) {
      emit("{} = -{}({});", var_decl, dt_name, operand_name);
    } else if (stmt->op_type == UnaryOpType::rsqrt) {
      emit("{} = {}(inversesqrt({}));", var_decl, dt_name, operand_name);
    } else if (stmt->op_type == UnaryOpType::sgn) {
      emit("{} = {}(sign({}));", var_decl, dt_name, operand_name);
    } else if (stmt->op_type == UnaryOpType::bit_not) {
      emit("{} = ~{}({});", var_decl, dt_name, operand_name);
    } else if (stmt->op_type == UnaryOpType::cast_value) {
      emit("{} = {}({});", var_decl, dt_name, operand_name);
    } else if (stmt->op_type == UnaryOpType::cast_bits) {
      constexpr int kFloatingPoint = 0;
      constexpr int kSignedInteger = 1;
      constexpr int kUnsignedInteger = 2;

      const auto dst_type = stmt->cast_type;
      const auto src_type = stmt->operand->element_type();
      auto dst_type_id = kFloatingPoint;
      if (is_integral(dst_type)) {
        dst_type_id = is_unsigned(dst_type) ? kUnsignedInteger : kSignedInteger;
      }
      auto src_type_id = kFloatingPoint;
      if (is_integral(src_type)) {
        src_type_id = is_unsigned(src_type) ? kUnsignedInteger : kSignedInteger;
      }

      TI_ASSERT_INFO(
          data_type_size(dst_type) == data_type_size(src_type),
          "bit_cast is only supported between data type with same size");

      if (dst_type_id != kFloatingPoint && src_type_id != kFloatingPoint) {
        emit("{} = {}({});", var_decl, dt_name, operand_name);
      } else if (dst_type_id == kFloatingPoint &&
                 src_type_id == kSignedInteger) {
        emit("{} = intBitsToFloat({});", var_decl, operand_name);
      } else if (dst_type_id == kSignedInteger &&
                 src_type_id == kFloatingPoint) {
        emit("{} = floatBitsToInt({});", var_decl, operand_name);
      } else if (dst_type_id == kFloatingPoint &&
                 src_type_id == kUnsignedInteger) {
        emit("{} = uintBitsToFloat({});", var_decl, operand_name);
      } else if (dst_type_id == kUnsignedInteger &&
                 src_type_id == kFloatingPoint) {
        emit("{} = floatBitsToUint({});", var_decl, operand_name);
      } else {
        TI_ERROR("[glsl] unsupported bit cast from {} to {}",
                 data_type_name(src_type), data_type_name(dst_type));
      }
    } else {
      emit("{} = {}({});", var_decl, unary_op_type_name(stmt->op_type),
           operand_name);
    }
  }

  void visit(BinaryOpStmt *bin) override {
    const auto dt_name = opengl_data_type_name(bin->element_type());
    const auto lhs_name = bin->lhs->raw_name();
    const auto rhs_name = bin->rhs->raw_name();
    const auto bin_name = bin->raw_name();
    const auto op_type = bin->op_type;
    const auto var_decl = fmt::format("const {} {}", dt_name, bin_name);
    if (op_type == BinaryOpType::floordiv) {
      if (is_integral(bin->lhs->element_type()) &&
          is_integral(bin->rhs->element_type())) {
        emit(
            "{} = {}(sign({}) * {} >= 0 ? abs({}) / abs({}) : "
            "sign({}) * "
            "(abs({}) + abs({}) - 1) / {});",
            var_decl, dt_name, lhs_name, rhs_name, lhs_name, rhs_name, lhs_name,
            lhs_name, rhs_name, rhs_name);
      } else {
        emit("{} = floor({} / {});", var_decl, lhs_name, rhs_name);
      }
      return;
    }
    if (bin->op_type == BinaryOpType::mod) {
      // NOTE: the GLSL built-in function `mod()` is a pythonic mod: x - y *
      // floor(x / y)
      emit("{} = {} - {} * int({} / {});", var_decl, lhs_name, rhs_name,
           lhs_name, rhs_name);
      return;
    }

    const auto binop = binary_op_type_symbol(bin->op_type);
    if (opengl::is_opengl_binary_op_infix(op_type)) {
      if (is_comparison(op_type)) {
        // TODO(#577): Taichi uses -1 as true due to LLVM i1.
        emit(" {} = -{}({} {} {});", var_decl, dt_name, lhs_name, binop,
             rhs_name);
      } else {
        emit("{} = {}({} {} {});", var_decl, dt_name, lhs_name, binop,
             rhs_name);
      }
    } else {
      // This is a function call
      emit("{} = {}({}, {});", var_decl, binop, lhs_name, rhs_name);
    }
  }

  void visit(TernaryOpStmt *tri) override {
    TI_ASSERT(tri->op_type == TernaryOpType::select);
    emit("const {} {} = ({}) ? ({}) : ({});",
         opengl_data_type_name(tri->element_type()), tri->raw_name(),
         tri->op1->raw_name(), tri->op2->raw_name(), tri->op3->raw_name());
  }

  void visit(AtomicOpStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    if (stmt->op_type != AtomicOpType::add) {
      TI_NOT_IMPLEMENTED;
    }
    const auto dt = stmt->dest->element_type().ptr_removed();
    std::string func = "atomicAdd";  // GLSL builtin
    std::string mem = at_buffer(stmt->dest, dt);
    if (dt->is_primitive(PrimitiveTypeID::f32)) {
      // Buffer has to be specified in the fatomicAdd helpers.
      const std::string buffer_name =
          buffer_instance_name(ptr_to_buffers_.at(stmt->dest));
      func = fmt::format("fatomicAdd_{}", buffer_name);
      mem = vk_data_address_shifter(stmt->dest, dt);
    } else if (!is_integral(dt)) {
      TI_ERROR("Vulkan only supports 32-bit atomic data types");
    }
    // const dt stmt = atomicAdd(mem, val);
    emit("const {} {} = {}({}, {});", opengl_data_type_name(dt),
         stmt->raw_name(), func, mem, stmt->val->raw_name());
  }

  void visit(IfStmt *if_stmt) override {
    emit("if ({} != 0) {{", if_stmt->cond->raw_name());
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

  void visit(WhileStmt *stmt) override {
    emit("while (true) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(WhileControlStmt *stmt) override {
    emit("if ({} == 0) break;", stmt->cond->raw_name());
  }

  void visit(ContinueStmt *stmt) override {
    if (stmt->as_return()) {
      emit("return;");
    } else {
      emit("continue;");
    }
  }

 private:
  void emit_headers() {
    SectionGuard sg(this, Section::Headers);

    emit("#version 450");
    emit("layout(local_size_x={}, local_size_y=1, local_size_z=1) in;",
         task_attribs_.advisory_num_threads_per_group);
    emit("");
    for (const auto &bb : task_attribs_.buffer_binds) {
      // e.g.
      // layout(std430, binding=0) buffer Root { int root_buffer[]; };
      emit("layout(std430, binding={}) buffer {} {{ int {}[]; }};", bb.binding,
           TaskAttributes::buffers_name(bb.type),
           buffer_instance_name(bb.type));
    }
    emit("");
    emit("// Helpers");
    current_appender().append_raw(kVulkanAtomicsSourceCode);
  }

  void generate_serial_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::serial;
    task_attribs_.buffer_binds = get_common_buffer_binds();
    task_attribs_.advisory_total_num_threads = 1;
    task_attribs_.advisory_num_threads_per_group = 1;

    const auto func_name = single_work_func_name();
    // The computation for a single work is wrapped inside a function, so that
    // we can do grid-strided loop.
    emit_single_work_func_def(func_name, stmt->body.get());
    // The actual compute kernel entry point.
    emit("void main() {{");
    {
      ScopedIndent s(current_appender());
      emit("// serial");
      emit("if ({} > 0) return;", kGlobalInvocationIDName);

      emit_call_single_work_func(func_name, /*loop_index_expr=*/"0");
    }
    // Close kernel
    emit("}}\n");
  }

  void generate_range_for_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::range_for;
    task_attribs_.buffer_binds = get_common_buffer_binds();

    task_attribs_.range_for_attribs = TaskAttributes::RangeForAttributes();
    auto &range_for_attribs = task_attribs_.range_for_attribs.value();
    range_for_attribs.const_begin = stmt->const_begin;
    range_for_attribs.const_end = stmt->const_end;
    range_for_attribs.begin =
        (stmt->const_begin ? stmt->begin_value : stmt->begin_offset);
    range_for_attribs.end =
        (stmt->const_end ? stmt->end_value : stmt->end_offset);

    const auto func_name = single_work_func_name();
    emit_single_work_func_def(func_name, stmt->body.get());

    emit("void main() {{");
    push_indent();
    const std::string total_elems_name("total_elems");
    std::string begin_expr;
    if (range_for_attribs.const_range()) {
      const int num_elems = range_for_attribs.end - range_for_attribs.begin;
      begin_expr = std::to_string(stmt->begin_value);
      emit("// range_for, range known at compile time");
      emit("const int {} = {};", total_elems_name, num_elems);
      task_attribs_.advisory_total_num_threads = num_elems;
    } else {
      TI_NOT_IMPLEMENTED;
    }
    // begin_ = thread_id   + begin_expr
    emit("const int begin_ = {} + {};", kGlobalInvocationIDName, begin_expr);
    // end_   = total_elems + begin_expr
    emit("const int end_ = {} + {};", total_elems_name, begin_expr);
    // For now, |total_invocs_name| is equal to |total_elems|. Once we support
    // dynamic range, they will be different.
    const std::string total_invocs_name = "total_invocs";
    // https://www.khronos.org/opengl/wiki/Compute_Shader#Inputs
    emit("const int {} = int(gl_NumWorkGroups.x * gl_WorkGroupSize.x);",
         total_invocs_name);
    // grid-strided loop
    emit("for (int ii = begin_; ii < end_; ii += {}) {{", total_invocs_name);
    {
      ScopedIndent s2(current_appender());
      emit_call_single_work_func(func_name, /*loop_index_expr=*/"ii");
    }
    emit("}}");  // closes for loop

    pop_indent();
    // Close kernel
    emit("}}\n");
    // TODO: runtime needs to verify if block_dim is feasible
    task_attribs_.advisory_num_threads_per_group = stmt->block_dim;
  }

  void emit_single_work_func_def(const std::string &func_name,

                                 Block *func_ir) {
    emit("void {}(", func_name);
    emit("    const int {}) {{", kLinearLoopIndexName);
    // We do not need additional indentation, because |func_ir| itself is a
    // block, which will be indented automatically.
    func_ir->accept(this);
    emit("}}\n");  // closes this function
  }

  void emit_call_single_work_func(const std::string &func_name,

                                  const std::string &loop_index_expr) {
    emit("{}({});", func_name, loop_index_expr);
  }

  std::string at_buffer(const Stmt *ptr, DataType dt) const {
    const std::string buffer_name =
        buffer_instance_name(ptr_to_buffers_.at(ptr));
    return fmt::format("{}[{}]", buffer_name, vk_data_address_shifter(ptr, dt));
  }

  std::string single_work_func_name() const {
    return task_name_ + "_func";
  }

  std::vector<BufferBind> get_common_buffer_binds() const {
    std::vector<BufferBind> result;
    int binding = 0;
    result.push_back({BuffersEnum::Root, binding++});
    result.push_back({BuffersEnum::GlobalTmps, binding++});
    if (!ctx_attribs_->empty()) {
      result.push_back({BuffersEnum::Context, binding++});
    }
    return result;
  }

  class SectionGuard {
   public:
    SectionGuard(TaskCodegen *tcg, Section new_sec)
        : tcg_(tcg), saved_(tcg->code_section_) {
      tcg_->code_section_ = new_sec;
    }

    ~SectionGuard() {
      tcg_->code_section_ = saved_;
    }

   private:
    TaskCodegen *const tcg_;
    const Section saved_;
  };

  friend class SectionGuard;

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    // TI_INFO(f, args...);
    current_appender().append(std::move(f), std::forward<Args>(args)...);
  }

  void push_indent() {
    current_appender().push_indent();
  }

  void pop_indent() {
    current_appender().pop_indent();
  }

  LineAppender &current_appender() {
    return section_appenders_[code_section_];
  }

  OffloadedStmt *const task_ir_;                        // not owned
  const CompiledSNodeStructs *const compiled_structs_;  // not owned
  const KernelContextAttributes *const ctx_attribs_;    // not owned
  const std::string task_name_;

  TaskAttributes task_attribs_;
  GetRootStmt *root_stmt_{nullptr};
  std::unordered_map<const Stmt *, BuffersEnum> ptr_to_buffers_;
  Section code_section_{Section::Kernels};
  std::unordered_map<Section, LineAppender> section_appenders_;
};

class KernelCodegen {
 public:
  struct Params {
    std::string ti_kernel_name;
    Kernel *kernel;
    const CompiledSNodeStructs *compiled_structs;
  };

  explicit KernelCodegen(const Params &params)
      : params_(params), ctx_attribs_(*params.kernel) {
  }

  using Result = VkRuntime::RegisterParams;

  Result run() {
    Result res;
    auto &kernel_attribs = res.kernel_attribs;
    auto *root = params_.kernel->ir->as<Block>();
    auto &tasks = root->statements;
    for (int i = 0; i < tasks.size(); ++i) {
      TaskCodegen::Params tp;
      tp.task_ir = tasks[i]->as<OffloadedStmt>();
      tp.task_id_in_kernel = i;
      tp.compiled_structs = params_.compiled_structs;
      tp.ctx_attribs = &ctx_attribs_;
      tp.ti_kernel_name = params_.ti_kernel_name;

      TaskCodegen cgen(tp);
      auto task_res = cgen.run();
      kernel_attribs.tasks_attribs.push_back(std::move(task_res.task_attribs));
      res.task_glsl_source_codes.push_back(std::move(task_res.source_code));
    }
    kernel_attribs.ctx_attribs = std::move(ctx_attribs_);
    kernel_attribs.name = params_.ti_kernel_name;
    kernel_attribs.is_jit_evaluator = params_.kernel->is_evaluator;
    return res;
  }

 private:
  Params params_;
  KernelContextAttributes ctx_attribs_;
};

}  // namespace

void lower(Kernel *kernel) {
  auto &config = kernel->program->config;
  config.demote_dense_struct_fors = true;
  irpass::compile_to_executable(kernel->ir.get(), config, kernel,
                                /*vectorize=*/false, kernel->grad,
                                /*ad_use_stack=*/false, config.print_ir,
                                /*lower_global_access=*/true,
                                /*make_thread_local=*/false);
}

FunctionType compile_to_executable(Kernel *kernel,
                                   const CompiledSNodeStructs *compiled_structs,
                                   VkRuntime *runtime) {
  const auto id = Program::get_kernel_id();
  const auto taichi_kernel_name(fmt::format("{}_k{:04d}_vk", kernel->name, id));
  TI_INFO("VK codegen for Taichi kernel={}", taichi_kernel_name);
  KernelCodegen::Params params;
  params.ti_kernel_name = taichi_kernel_name;
  params.kernel = kernel;
  params.compiled_structs = compiled_structs;
  KernelCodegen codegen(params);
  auto res = codegen.run();
  auto handle = runtime->register_taichi_kernel(std::move(res));
  return [runtime, handle, taichi_kernel_name](Context &ctx) {
    runtime->launch_kernel(handle, &ctx);
  };
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
