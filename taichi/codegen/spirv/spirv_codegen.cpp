#include "taichi/codegen/spirv/spirv_codegen.h"

#include <string>
#include <vector>
#include <variant>

#include "taichi/codegen/codegen_utils.h"
#include "taichi/program/program.h"
#include "taichi/program/kernel.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/ir.h"
#include "taichi/util/line_appender.h"
#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/codegen/spirv/spirv_ir_builder.h"
#include "taichi/ir/transforms.h"
#include "taichi/math/arithmetic.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>
#include "fp16.h"

namespace taichi::lang {
namespace spirv {
namespace {

constexpr char kRootBufferName[] = "root_buffer";
constexpr char kGlobalTmpsBufferName[] = "global_tmps_buffer";
constexpr char kArgsBufferName[] = "args_buffer";
constexpr char kRetBufferName[] = "ret_buffer";
constexpr char kListgenBufferName[] = "listgen_buffer";
constexpr char kExtArrBufferName[] = "ext_arr_buffer";

constexpr int kMaxNumThreadsGridStrideLoop = 65536 * 2;

using BufferType = TaskAttributes::BufferType;
using BufferInfo = TaskAttributes::BufferInfo;
using BufferBind = TaskAttributes::BufferBind;
using BufferInfoHasher = TaskAttributes::BufferInfoHasher;

using TextureBind = TaskAttributes::TextureBind;

std::string buffer_instance_name(BufferInfo b) {
  // https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Syntax
  switch (b.type) {
    case BufferType::Root:
      return std::string(kRootBufferName) + "_" + std::to_string(b.root_id);
    case BufferType::GlobalTmps:
      return kGlobalTmpsBufferName;
    case BufferType::Args:
      return kArgsBufferName;
    case BufferType::Rets:
      return kRetBufferName;
    case BufferType::ListGen:
      return kListgenBufferName;
    case BufferType::ExtArr:
      return std::string(kExtArrBufferName) + "_" + std::to_string(b.root_id);
    default:
      TI_NOT_IMPLEMENTED;
      break;
  }
  return {};
}

class TaskCodegen : public IRVisitor {
 public:
  struct Params {
    OffloadedStmt *task_ir;
    Arch arch;
    DeviceCapabilityConfig *caps;
    std::vector<CompiledSNodeStructs> compiled_structs;
    const KernelContextAttributes *ctx_attribs;
    std::string ti_kernel_name;
    int task_id_in_kernel;
  };

  const bool use_64bit_pointers = false;

  explicit TaskCodegen(const Params &params)
      : arch_(params.arch),
        caps_(params.caps),
        task_ir_(params.task_ir),
        compiled_structs_(params.compiled_structs),
        ctx_attribs_(params.ctx_attribs),
        task_name_(fmt::format("{}_t{:02d}",
                               params.ti_kernel_name,
                               params.task_id_in_kernel)) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;

    fill_snode_to_root();
    ir_ = std::make_shared<spirv::IRBuilder>(arch_, caps_);
  }

  void fill_snode_to_root() {
    for (int root = 0; root < compiled_structs_.size(); ++root) {
      for (auto &[node_id, node] : compiled_structs_[root].snode_descriptors) {
        snode_to_root_[node_id] = root;
      }
    }
  }

  // Replace the wild '%' in the format string with "%%".
  std::string sanitize_format_string(std::string const &str) {
    std::string sanitized_str;
    for (char c : str) {
      if (c == '%') {
        sanitized_str += "%%";
      } else {
        sanitized_str += c;
      }
    }
    return sanitized_str;
  }

  struct Result {
    std::vector<uint32_t> spirv_code;
    TaskAttributes task_attribs;
    std::unordered_map<int, irpass::ExternalPtrAccess> arr_access;
  };

  Result run() {
    ir_->init_header();
    kernel_function_ = ir_->new_function();  // void main();
    ir_->debug_name(spv::OpName, kernel_function_, "main");

    if (task_ir_->task_type == OffloadedTaskType::serial) {
      generate_serial_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::range_for) {
      // struct_for is automatically lowered to ranged_for for dense snodes
      generate_range_for_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::struct_for) {
      generate_struct_for_kernel(task_ir_);
    } else {
      TI_ERROR("Unsupported offload type={} on SPIR-V codegen",
               task_ir_->task_name());
    }
    // Headers need global information, so it has to be delayed after visiting
    // the task IR.
    emit_headers();

    Result res;
    res.spirv_code = ir_->finalize();
    res.task_attribs = std::move(task_attribs_);
    res.arr_access = irpass::detect_external_ptr_access_in_task(task_ir_);

    return res;
  }

  void visit(OffloadedStmt *) override {
    TI_ERROR("This codegen is supposed to deal with one offloaded task");
  }

  void visit(Block *stmt) override {
    for (auto &s : stmt->statements) {
      if (offload_loop_motion_.find(s.get()) == offload_loop_motion_.end()) {
        s->accept(this);
      }
    }
  }

  void visit(PrintStmt *stmt) override {
    if (!caps_->get(DeviceCapability::spirv_has_non_semantic_info)) {
      return;
    }

    std::string formats;
    std::vector<Value> vals;

    for (auto i = 0; i < stmt->contents.size(); ++i) {
      auto const &content = stmt->contents[i];
      auto const &format = stmt->formats[i];
      if (std::holds_alternative<Stmt *>(content)) {
        auto arg_stmt = std::get<Stmt *>(content);
        TI_ASSERT(!arg_stmt->ret_type->is<TensorType>());

        auto value = ir_->query_value(arg_stmt->raw_name());
        vals.push_back(value);

        auto &&merged_format = merge_printf_specifier(
            format, data_type_format(arg_stmt->ret_type));
        // Vulkan doesn't support length, flags, or width specifier, except for
        // unsigned long.
        // https://vulkan.lunarg.com/doc/view/1.3.204.1/windows/debug_printf.html
        auto &&[format_flags, format_width, format_precision, format_length,
                format_conversion] = parse_printf_specifier(merged_format);
        if (!format_flags.empty()) {
          TI_WARN(
              "The printf flags '{}' are not supported in Vulkan, "
              "and will be discarded.",
              format_flags);
          format_flags.clear();
        }
        if (!format_width.empty()) {
          TI_WARN(
              "The printf width modifier '{}' is not supported in Vulkan, "
              "and will be discarded.",
              format_width);
          format_width.clear();
        }
        if (!format_length.empty() &&
            !(format_length == "l" &&
              (format_conversion == "u" || format_conversion == "x"))) {
          TI_WARN(
              "The printf length modifier '{}' is not supported in Vulkan, "
              "and will be discarded.",
              format_length);
          format_length.clear();
        }
        formats +=
            "%" +
            format_precision.append(format_length).append(format_conversion);
      } else {
        auto arg_str = std::get<std::string>(content);
        formats += sanitize_format_string(arg_str);
      }
    }
    ir_->call_debugprintf(formats, vals);
  }

  void visit(ConstStmt *const_stmt) override {
    auto get_const = [&](const TypedConstant &const_val) {
      auto dt = const_val.dt.ptr_removed();
      spirv::SType stype = ir_->get_primitive_type(dt);

      if (dt->is_primitive(PrimitiveTypeID::f32)) {
        return ir_->float_immediate_number(
            stype, static_cast<double>(const_val.val_f32), false);
      } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
        // Ref: See taichi::lang::TypedConstant::TypedConstant()
        // FP16 is stored as FP32 on host side,
        // as some CPUs does not have native FP16 (and no libc support)
        return ir_->float_immediate_number(
            stype, static_cast<double>(const_val.val_f32), false);
      } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
        return ir_->int_immediate_number(
            stype, static_cast<int64_t>(const_val.val_i32), false);
      } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
        return ir_->int_immediate_number(
            stype, static_cast<int64_t>(const_val.val_i64), false);
      } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
        return ir_->float_immediate_number(
            stype, static_cast<double>(const_val.val_f64), false);
      } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
        return ir_->int_immediate_number(
            stype, static_cast<int64_t>(const_val.val_i8), false);
      } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
        return ir_->int_immediate_number(
            stype, static_cast<int64_t>(const_val.val_i16), false);
      } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
        return ir_->uint_immediate_number(
            stype, static_cast<uint64_t>(const_val.val_u8), false);
      } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
        return ir_->uint_immediate_number(
            stype, static_cast<uint64_t>(const_val.val_u16), false);
      } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
        return ir_->uint_immediate_number(
            stype, static_cast<uint64_t>(const_val.val_u32), false);
      } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
        return ir_->uint_immediate_number(
            stype, static_cast<uint64_t>(const_val.val_u64), false);
      } else {
        TI_P(data_type_name(dt));
        TI_NOT_IMPLEMENTED
        return spirv::Value();
      }
    };

    spirv::Value val = get_const(const_stmt->val);
    ir_->register_value(const_stmt->raw_name(), val);
  }

  void visit(AllocaStmt *alloca) override {
    spirv::Value ptr_val;
    if (alloca->ret_type->is<TensorType>()) {
      auto tensor_type = alloca->ret_type->cast<TensorType>();
      auto elem_num = tensor_type->get_num_elements();
      spirv::SType elem_type =
          ir_->get_primitive_type(tensor_type->get_element_type());
      spirv::SType arr_type = ir_->get_array_type(elem_type, elem_num);
      if (alloca->is_shared) {  // for shared memory / workgroup memory
        ptr_val = ir_->alloca_workgroup_array(arr_type);
        shared_array_binds_.push_back(ptr_val);
      } else {  // for function memory
        ptr_val = ir_->alloca_variable(arr_type);
      }
    } else {
      // Alloca for a single variable
      spirv::SType src_type = ir_->get_primitive_type(alloca->element_type());
      ptr_val = ir_->alloca_variable(src_type);
      ir_->store_variable(ptr_val, ir_->get_zero(src_type));
    }
    ir_->register_value(alloca->raw_name(), ptr_val);
  }

  void visit(MatrixPtrStmt *stmt) override {
    spirv::Value ptr_val;
    spirv::Value origin_val = ir_->query_value(stmt->origin->raw_name());
    spirv::Value offset_val = ir_->query_value(stmt->offset->raw_name());
    auto dt = stmt->element_type().ptr_removed();
    if (stmt->offset_used_as_index()) {
      if (stmt->origin->is<AllocaStmt>()) {
        spirv::SType ptr_type = ir_->get_pointer_type(
            ir_->get_primitive_type(dt), origin_val.stype.storage_class);
        ptr_val = ir_->make_value(spv::OpAccessChain, ptr_type, origin_val,
                                  offset_val);
        if (stmt->origin->as<AllocaStmt>()->is_shared) {
          ptr_to_buffers_[stmt] = ptr_to_buffers_[stmt->origin];
        }
      } else if (stmt->origin->is<GlobalTemporaryStmt>()) {
        spirv::Value dt_bytes = ir_->int_immediate_number(
            ir_->i32_type(), ir_->get_primitive_type_size(dt), false);
        spirv::Value offset_bytes = ir_->mul(dt_bytes, offset_val);
        ptr_val = ir_->add(origin_val, offset_bytes);
        ptr_to_buffers_[stmt] = ptr_to_buffers_[stmt->origin];
      } else {
        TI_NOT_IMPLEMENTED;
      }
    } else {  // offset used as bytes
      ptr_val = ir_->add(origin_val, ir_->cast(origin_val.stype, offset_val));
      ptr_to_buffers_[stmt] = ptr_to_buffers_[stmt->origin];
    }
    ir_->register_value(stmt->raw_name(), ptr_val);
  }

  void visit(LocalLoadStmt *stmt) override {
    auto ptr = stmt->src;
    spirv::Value ptr_val = ir_->query_value(ptr->raw_name());
    spirv::Value val = ir_->load_variable(
        ptr_val, ir_->get_primitive_type(stmt->element_type()));
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(LocalStoreStmt *stmt) override {
    spirv::Value ptr_val = ir_->query_value(stmt->dest->raw_name());
    spirv::Value val = ir_->query_value(stmt->val->raw_name());
    ir_->store_variable(ptr_val, val);
  }

  void visit(GetRootStmt *stmt) override {
    const int root_id = snode_to_root_.at(stmt->root()->id);
    root_stmts_[root_id] = stmt;
    // get_buffer_value({BufferType::Root, root_id}, PrimitiveType::u32);
    spirv::Value root_val = make_pointer(0);
    ir_->register_value(stmt->raw_name(), root_val);
  }

  void visit(GetChStmt *stmt) override {
    // TODO: GetChStmt -> GetComponentStmt ?
    const int root = snode_to_root_.at(stmt->input_snode->id);

    const auto &snode_descs = compiled_structs_[root].snode_descriptors;
    auto *out_snode = stmt->output_snode;
    TI_ASSERT(snode_descs.at(stmt->input_snode->id).get_child(stmt->chid) ==
              out_snode);

    const auto &desc = snode_descs.at(out_snode->id);

    spirv::Value input_ptr_val = ir_->query_value(stmt->input_ptr->raw_name());
    spirv::Value offset = make_pointer(desc.mem_offset_in_parent_cell);
    spirv::Value val = ir_->add(input_ptr_val, offset);
    ir_->register_value(stmt->raw_name(), val);

    if (out_snode->is_place()) {
      TI_ASSERT(ptr_to_buffers_.count(stmt) == 0);
      ptr_to_buffers_[stmt] = BufferInfo(BufferType::Root, root);
    }
  }

  enum class ActivationOp { activate, deactivate, query };

  spirv::Value bitmasked_activation(ActivationOp op,
                                    spirv::Value parent_ptr,
                                    int root_id,
                                    const SNode *sn,
                                    spirv::Value input_index) {
    spirv::SType ptr_dt = parent_ptr.stype;
    const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
    const auto &desc = snode_descs.at(sn->id);

    auto bitmask_word_index =
        ir_->make_value(spv::OpShiftRightLogical, ptr_dt, input_index,
                        ir_->uint_immediate_number(ptr_dt, 5));
    auto bitmask_bit_index =
        ir_->make_value(spv::OpBitwiseAnd, ptr_dt, input_index,
                        ir_->uint_immediate_number(ptr_dt, 31));
    auto bitmask_mask = ir_->make_value(spv::OpShiftLeftLogical, ptr_dt,
                                        ir_->const_i32_one_, bitmask_bit_index);

    auto buffer = get_buffer_value(BufferInfo(BufferType::Root, root_id),
                                   PrimitiveType::u32);
    auto bitmask_word_ptr =
        ir_->make_value(spv::OpShiftLeftLogical, ptr_dt, bitmask_word_index,
                        ir_->uint_immediate_number(ir_->u32_type(), 2));
    bitmask_word_ptr = ir_->add(
        bitmask_word_ptr,
        make_pointer(desc.cell_stride * desc.snode->num_cells_per_container));
    bitmask_word_ptr = ir_->add(parent_ptr, bitmask_word_ptr);
    bitmask_word_ptr = ir_->make_value(
        spv::OpShiftRightLogical, ir_->u32_type(), bitmask_word_ptr,
        ir_->uint_immediate_number(ir_->u32_type(), 2));
    bitmask_word_ptr =
        ir_->struct_array_access(ir_->u32_type(), buffer, bitmask_word_ptr);

    if (op == ActivationOp::activate) {
      return ir_->make_value(spv::OpAtomicOr, ir_->u32_type(), bitmask_word_ptr,
                             /*scope=*/ir_->const_i32_one_,
                             /*semantics=*/ir_->const_i32_zero_, bitmask_mask);
    } else if (op == ActivationOp::deactivate) {
      bitmask_mask = ir_->make_value(spv::OpNot, ir_->u32_type(), bitmask_mask);
      return ir_->make_value(spv::OpAtomicAnd, ir_->u32_type(),
                             bitmask_word_ptr,
                             /*scope=*/ir_->const_i32_one_,
                             /*semantics=*/ir_->const_i32_zero_, bitmask_mask);
    } else {
      auto bitmask_val = ir_->load_variable(bitmask_word_ptr, ir_->u32_type());
      auto bit = ir_->make_value(spv::OpShiftRightLogical, ir_->u32_type(),
                                 bitmask_val, bitmask_bit_index);
      bit = ir_->make_value(spv::OpBitwiseAnd, ir_->u32_type(), bit,
                            ir_->uint_immediate_number(ir_->u32_type(), 1));
      return ir_->make_value(spv::OpUGreaterThan, ir_->bool_type(), bit,
                             ir_->uint_immediate_number(ir_->u32_type(), 0));
    }
  }

  void visit(SNodeOpStmt *stmt) override {
    const int root_id = snode_to_root_.at(stmt->snode->id);
    std::string parent = stmt->ptr->raw_name();
    spirv::Value parent_val = ir_->query_value(parent);

    if (stmt->snode->type == SNodeType::bitmasked) {
      spirv::Value input_index_val =
          ir_->cast(parent_val.stype, ir_->query_value(stmt->val->raw_name()));

      if (stmt->op_type == SNodeOpType::is_active) {
        auto is_active =
            bitmasked_activation(ActivationOp::query, parent_val, root_id,
                                 stmt->snode, input_index_val);
        is_active =
            ir_->cast(ir_->get_primitive_type(stmt->ret_type), is_active);
        is_active = ir_->make_value(spv::OpSNegate, is_active.stype, is_active);
        ir_->register_value(stmt->raw_name(), is_active);
      } else if (stmt->op_type == SNodeOpType::deactivate) {
        bitmasked_activation(ActivationOp::deactivate, parent_val, root_id,
                             stmt->snode, input_index_val);
      } else if (stmt->op_type == SNodeOpType::activate) {
        bitmasked_activation(ActivationOp::activate, parent_val, root_id,
                             stmt->snode, input_index_val);
      } else {
        TI_NOT_IMPLEMENTED;
      }
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(SNodeLookupStmt *stmt) override {
    // TODO: SNodeLookupStmt -> GetSNodeCellStmt ?
    bool is_root{false};  // Eliminate first root snode access
    const int root_id = snode_to_root_.at(stmt->snode->id);
    std::string parent;

    if (stmt->input_snode) {
      parent = stmt->input_snode->raw_name();
    } else {
      TI_ASSERT(root_stmts_.at(root_id) != nullptr);
      parent = root_stmts_.at(root_id)->raw_name();
    }
    const auto *sn = stmt->snode;

    spirv::Value parent_val = ir_->query_value(parent);

    if (stmt->activate) {
      if (sn->type == SNodeType::dense) {
        // Do nothing
      } else if (sn->type == SNodeType::bitmasked) {
        spirv::Value input_index_val =
            ir_->query_value(stmt->input_index->raw_name());
        bitmasked_activation(ActivationOp::activate, parent_val, root_id, sn,
                             input_index_val);
      } else {
        TI_NOT_IMPLEMENTED;
      }
    }

    spirv::Value val;
    if (is_root) {
      val = parent_val;  // Assert Root[0] access at first time
    } else {
      const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
      const auto &desc = snode_descs.at(sn->id);

      spirv::Value input_index_val = ir_->cast(
          parent_val.stype, ir_->query_value(stmt->input_index->raw_name()));
      spirv::Value stride = make_pointer(desc.cell_stride);
      spirv::Value offset = ir_->mul(input_index_val, stride);
      val = ir_->add(parent_val, offset);
    }
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(RandStmt *stmt) override {
    spirv::Value val;
    spirv::Value global_tmp =
        get_buffer_value(BufferType::GlobalTmps, PrimitiveType::u32);
    if (stmt->element_type()->is_primitive(PrimitiveTypeID::i32)) {
      val = ir_->rand_i32(global_tmp);
    } else if (stmt->element_type()->is_primitive(PrimitiveTypeID::u32)) {
      val = ir_->rand_u32(global_tmp);
    } else if (stmt->element_type()->is_primitive(PrimitiveTypeID::f32)) {
      val = ir_->rand_f32(global_tmp);
    } else if (stmt->element_type()->is_primitive(PrimitiveTypeID::f16)) {
      auto highp_val = ir_->rand_f32(global_tmp);
      val = ir_->cast(ir_->f16_type(), highp_val);
    } else {
      TI_ERROR("rand only support 32-bit type");
    }
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(LinearizeStmt *stmt) override {
    spirv::Value val = ir_->const_i32_zero_;
    for (size_t i = 0; i < stmt->inputs.size(); ++i) {
      spirv::Value strides_val =
          ir_->int_immediate_number(ir_->i32_type(), stmt->strides[i]);
      spirv::Value input_val = ir_->query_value(stmt->inputs[i]->raw_name());
      val = ir_->add(ir_->mul(val, strides_val), input_val);
    }
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(LoopIndexStmt *stmt) override {
    const auto stmt_name = stmt->raw_name();
    if (stmt->loop->is<OffloadedStmt>()) {
      const auto type = stmt->loop->as<OffloadedStmt>()->task_type;
      if (type == OffloadedTaskType::range_for) {
        TI_ASSERT(stmt->index == 0);
        spirv::Value loop_var = ir_->query_value("ii");
        // spirv::Value val = ir_->add(loop_var, ir_->const_i32_zero_);
        ir_->register_value(stmt_name, loop_var);
      } else {
        TI_NOT_IMPLEMENTED;
      }
    } else if (stmt->loop->is<RangeForStmt>()) {
      TI_ASSERT(stmt->index == 0);
      spirv::Value loop_var = ir_->query_value(stmt->loop->raw_name());
      spirv::Value val = ir_->add(loop_var, ir_->const_i32_zero_);
      ir_->register_value(stmt_name, val);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    spirv::Value val = ir_->query_value(stmt->val->raw_name());

    store_buffer(stmt->dest, val);
  }

  void visit(GlobalLoadStmt *stmt) override {
    auto dt = stmt->element_type();

    auto val = load_buffer(stmt->src, dt);

    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(ArgLoadStmt *stmt) override {
    const auto arg_id = stmt->arg_id;
    const auto arg_type = ctx_attribs_->args_type()->get_element_type({arg_id});
    if (arg_type->is<PointerType>() ||
        (arg_type->is<lang::StructType>() &&
         arg_type->as<lang::StructType>()->elements().size() >= 2 &&
         arg_type->as<lang::StructType>()
             ->get_element_type({1})
             ->is<PointerType>())) {
      // Do not shift! We are indexing the buffers at byte granularity.
      // spirv::Value val =
      //    ir_->int_immediate_number(ir_->i32_type(), offset_in_mem);
      // ir_->register_value(stmt->raw_name(), val);
    } else {
      auto buffer_value =
          get_buffer_value(BufferType::Args, PrimitiveType::i32);
      const auto val_type = args_struct_types_.at(arg_id);
      spirv::Value buffer_val = ir_->make_value(
          spv::OpAccessChain,
          ir_->get_pointer_type(val_type, spv::StorageClassUniform),
          buffer_value, ir_->int_immediate_number(ir_->i32_type(), arg_id));
      buffer_val.flag = ValueKind::kVariablePtr;
      if (!stmt->create_load) {
        ir_->register_value(stmt->raw_name(), buffer_val);
        return;
      }
      spirv::Value val = ir_->load_variable(buffer_val, val_type);
      ir_->register_value(stmt->raw_name(), val);
    }
  }

  void visit(GetElementStmt *stmt) override {
    spirv::Value val = ir_->query_value(stmt->src->raw_name());
    const auto val_type = ir_->get_primitive_type(stmt->element_type());
    const auto val_type_ptr =
        ir_->get_pointer_type(val_type, spv::StorageClassUniform);
    val = ir_->make_access_chain(val_type_ptr, val, stmt->index);
    val = ir_->load_variable(val, val_type);
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(ReturnStmt *stmt) override {
    // Now we only support one ret
    auto dt = stmt->element_types()[0];
    for (int i = 0; i < stmt->values.size(); i++) {
      spirv::Value buffer_val = ir_->make_value(
          spv::OpAccessChain,
          ir_->get_storage_pointer_type(ir_->get_primitive_type(dt)),
          get_buffer_value(BufferType::Rets, dt),
          ir_->int_immediate_number(ir_->i32_type(), 0),
          ir_->int_immediate_number(ir_->i32_type(), i));
      buffer_val.flag = ValueKind::kVariablePtr;
      spirv::Value val = ir_->query_value(stmt->values[i]->raw_name());
      ir_->store_variable(buffer_val, val);
    }
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    spirv::Value val = ir_->int_immediate_number(ir_->i32_type(), stmt->offset,
                                                 false);  // Named Constant
    ir_->register_value(stmt->raw_name(), val);
    ptr_to_buffers_[stmt] = BufferType::GlobalTmps;
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    const auto name = stmt->raw_name();
    const auto arg_id = stmt->arg_id;
    const auto axis = stmt->axis;

    const auto extra_args_member_index = ctx_attribs_->args().size();

    const auto extra_arg_index = (arg_id * taichi_max_num_indices) + axis;
    spirv::Value var_ptr;
    if (ctx_attribs_->args_type()
            ->get_element_type({arg_id})
            ->is<lang::StructType>()) {
      // Is ndarray
      var_ptr = ir_->make_value(
          spv::OpAccessChain,
          ir_->get_pointer_type(ir_->i32_type(), spv::StorageClassUniform),
          get_buffer_value(BufferType::Args, PrimitiveType::i32),
          ir_->int_immediate_number(ir_->i32_type(), arg_id),
          ir_->int_immediate_number(ir_->i32_type(), 0),
          ir_->int_immediate_number(ir_->i32_type(), axis));
    } else {
      // Is texture
      var_ptr = ir_->make_value(
          spv::OpAccessChain,
          ir_->get_pointer_type(ir_->i32_type(), spv::StorageClassUniform),
          get_buffer_value(BufferType::Args, PrimitiveType::i32),
          ir_->int_immediate_number(ir_->i32_type(),
                                    extra_args_member_index + extra_arg_index));
    }
    spirv::Value var = ir_->load_variable(var_ptr, ir_->i32_type());

    ir_->register_value(name, var);
  }

  void visit(ExternalPtrStmt *stmt) override {
    // Used mostly for transferring data between host (e.g. numpy array) and
    // device.
    spirv::Value linear_offset = ir_->int_immediate_number(ir_->i32_type(), 0);
    const auto *argload = stmt->base_ptr->as<ArgLoadStmt>();
    const int arg_id = argload->arg_id;
    {
      const int num_indices = stmt->indices.size();
      std::vector<std::string> size_var_names;
      const auto &element_shape = stmt->element_shape;
      const auto layout = stmt->element_dim <= 0 ? ExternalArrayLayout::kAOS
                                                 : ExternalArrayLayout::kSOA;
      const size_t element_shape_index_offset =
          (layout == ExternalArrayLayout::kAOS)
              ? num_indices - element_shape.size()
              : 0;
      for (int i = 0; i < num_indices - element_shape.size(); i++) {
        std::string var_name = fmt::format("{}_size{}_", stmt->raw_name(), i);
        spirv::Value var_ptr = ir_->make_value(
            spv::OpAccessChain,
            ir_->get_pointer_type(ir_->i32_type(), spv::StorageClassUniform),
            get_buffer_value(BufferType::Args, PrimitiveType::i32),
            ir_->int_immediate_number(ir_->i32_type(), arg_id),
            ir_->int_immediate_number(ir_->i32_type(), 0),
            ir_->int_immediate_number(ir_->i32_type(), i));
        spirv::Value var = ir_->load_variable(var_ptr, ir_->i32_type());
        ir_->register_value(var_name, var);
        size_var_names.push_back(std::move(var_name));
      }
      int size_var_names_idx = 0;
      for (int i = 0; i < num_indices; i++) {
        spirv::Value size_var;
        // Use immediate numbers to flatten index for element shapes.
        if (i >= element_shape_index_offset &&
            i < element_shape_index_offset + element_shape.size()) {
          size_var = ir_->uint_immediate_number(
              ir_->i32_type(), element_shape[i - element_shape_index_offset]);
        } else {
          size_var = ir_->query_value(size_var_names[size_var_names_idx++]);
        }
        spirv::Value indices = ir_->query_value(stmt->indices[i]->raw_name());
        linear_offset = ir_->mul(linear_offset, size_var);
        linear_offset = ir_->add(linear_offset, indices);
      }
      linear_offset = ir_->make_value(
          spv::OpShiftLeftLogical, ir_->i32_type(), linear_offset,
          ir_->int_immediate_number(ir_->i32_type(),
                                    log2int(ir_->get_primitive_type_size(
                                        stmt->ret_type.ptr_removed()))));
      if (caps_->get(DeviceCapability::spirv_has_no_integer_wrap_decoration)) {
        ir_->decorate(spv::OpDecorate, linear_offset,
                      spv::DecorationNoSignedWrap);
      }
    }
    if (caps_->get(DeviceCapability::spirv_has_physical_storage_buffer)) {
      spirv::Value addr_ptr = ir_->make_value(
          spv::OpAccessChain,
          ir_->get_pointer_type(ir_->u64_type(), spv::StorageClassUniform),
          get_buffer_value(BufferType::Args, PrimitiveType::i32),
          ir_->int_immediate_number(ir_->i32_type(), arg_id),
          ir_->int_immediate_number(ir_->i32_type(), 1));
      spirv::Value addr = ir_->load_variable(addr_ptr, ir_->u64_type());
      addr = ir_->add(addr, ir_->make_value(spv::OpSConvert, ir_->u64_type(),
                                            linear_offset));
      ir_->register_value(stmt->raw_name(), addr);
    } else {
      ir_->register_value(stmt->raw_name(), linear_offset);
    }

    if (ctx_attribs_->args()[arg_id].is_array) {
      ptr_to_buffers_[stmt] = {BufferType::ExtArr, arg_id};
    } else {
      ptr_to_buffers_[stmt] = BufferType::Args;
    }
  }

  void visit(DecorationStmt *stmt) override {
  }

  void visit(UnaryOpStmt *stmt) override {
    const auto operand_name = stmt->operand->raw_name();

    const auto src_dt = stmt->operand->element_type();
    const auto dst_dt = stmt->element_type();
    spirv::SType src_type = ir_->get_primitive_type(src_dt);
    spirv::SType dst_type;
    if (dst_dt.is_pointer()) {
      auto stype = dst_dt.ptr_removed()->as<lang::StructType>();
      std::vector<std::tuple<SType, std::string, size_t>> components;
      for (int i = 0; i < stype->get_num_elements(); i++) {
        components.push_back(
            {ir_->get_primitive_type(stype->get_element_type({i})),
             fmt::format("element{}", i), stype->get_element_offset({i})});
      }
      dst_type = ir_->create_struct_type(components);
    } else {
      dst_type = ir_->get_primitive_type(dst_dt);
    }
    spirv::Value operand_val = ir_->query_value(operand_name);
    spirv::Value val = spirv::Value();

    if (stmt->op_type == UnaryOpType::logic_not) {
      spirv::Value zero =
          ir_->get_zero(src_type);  // Math zero type to left hand side
      if (is_integral(src_dt)) {
        if (is_signed(src_dt)) {
          zero = ir_->int_immediate_number(src_type, 0);
        } else {
          zero = ir_->uint_immediate_number(src_type, 0);
        }
      } else if (is_real(src_dt)) {
        zero = ir_->float_immediate_number(src_type, 0);
      } else {
        TI_NOT_IMPLEMENTED
      }
      val = ir_->cast(dst_type, ir_->eq(operand_val, zero));
    } else if (stmt->op_type == UnaryOpType::neg) {
      operand_val = ir_->cast(dst_type, operand_val);
      if (is_integral(dst_dt)) {
        if (is_signed(dst_dt)) {
          val = ir_->make_value(spv::OpSNegate, dst_type, operand_val);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (is_real(dst_dt)) {
        val = ir_->make_value(spv::OpFNegate, dst_type, operand_val);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == UnaryOpType::rsqrt) {
      const uint32_t InverseSqrt_id = 32;
      if (is_real(src_dt)) {
        val = ir_->call_glsl450(src_type, InverseSqrt_id, operand_val);
        val = ir_->cast(dst_type, val);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == UnaryOpType::sgn) {
      const uint32_t FSign_id = 6;
      const uint32_t SSign_id = 7;
      if (is_integral(src_dt)) {
        if (is_signed(src_dt)) {
          val = ir_->call_glsl450(src_type, SSign_id, operand_val);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (is_real(src_dt)) {
        val = ir_->call_glsl450(src_type, FSign_id, operand_val);
      } else {
        TI_NOT_IMPLEMENTED
      }
      val = ir_->cast(dst_type, val);
    } else if (stmt->op_type == UnaryOpType::bit_not) {
      operand_val = ir_->cast(dst_type, operand_val);
      if (is_integral(dst_dt)) {
        val = ir_->make_value(spv::OpNot, dst_type, operand_val);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == UnaryOpType::cast_value) {
      val = ir_->cast(dst_type, operand_val);
    } else if (stmt->op_type == UnaryOpType::cast_bits) {
      if (data_type_bits(src_dt) == data_type_bits(dst_dt)) {
        val = ir_->make_value(spv::OpBitcast, dst_type, operand_val);
      } else {
        TI_ERROR("bit_cast is only supported between data type with same size");
      }
    } else if (stmt->op_type == UnaryOpType::abs) {
      const uint32_t FAbs_id = 4;
      const uint32_t SAbs_id = 5;
      if (src_type.id == dst_type.id) {
        if (is_integral(src_dt)) {
          if (is_signed(src_dt)) {
            val = ir_->call_glsl450(src_type, SAbs_id, operand_val);
          } else {
            TI_NOT_IMPLEMENTED
          }
        } else if (is_real(src_dt)) {
          val = ir_->call_glsl450(src_type, FAbs_id, operand_val);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == UnaryOpType::inv) {
      if (is_real(dst_dt)) {
        val = ir_->div(ir_->float_immediate_number(dst_type, 1), operand_val);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == UnaryOpType::frexp) {
      // FrexpStruct is the same type of the first member.
      val = ir_->alloca_variable(dst_type);
      auto v = ir_->call_glsl450(dst_type, 52, operand_val);
      ir_->store_variable(val, v);
    } else if (stmt->op_type == UnaryOpType::popcnt) {
      val = ir_->popcnt(operand_val);
    }
#define UNARY_OP_TO_SPIRV(op, instruction, instruction_id, max_bits)           \
  else if (stmt->op_type == UnaryOpType::op) {                                 \
    const uint32_t instruction = instruction_id;                               \
    if (is_real(src_dt)) {                                                     \
      if (data_type_bits(src_dt) > max_bits) {                                 \
        TI_ERROR("Instruction {}({}) does not {}bits operation", #instruction, \
                 instruction_id, data_type_bits(src_dt));                      \
      }                                                                        \
      val = ir_->call_glsl450(src_type, instruction, operand_val);             \
    } else {                                                                   \
      TI_NOT_IMPLEMENTED                                                       \
    }                                                                          \
  }
    UNARY_OP_TO_SPIRV(round, Round, 1, 64)
    UNARY_OP_TO_SPIRV(floor, Floor, 8, 64)
    UNARY_OP_TO_SPIRV(ceil, Ceil, 9, 64)
    UNARY_OP_TO_SPIRV(sin, Sin, 13, 32)
    UNARY_OP_TO_SPIRV(asin, Asin, 16, 32)
    UNARY_OP_TO_SPIRV(cos, Cos, 14, 32)
    UNARY_OP_TO_SPIRV(acos, Acos, 17, 32)
    UNARY_OP_TO_SPIRV(tan, Tan, 15, 32)
    UNARY_OP_TO_SPIRV(tanh, Tanh, 21, 32)
    UNARY_OP_TO_SPIRV(exp, Exp, 27, 32)
    UNARY_OP_TO_SPIRV(log, Log, 28, 32)
    UNARY_OP_TO_SPIRV(sqrt, Sqrt, 31, 64)
#undef UNARY_OP_TO_SPIRV
    else {TI_NOT_IMPLEMENTED} ir_->register_value(stmt->raw_name(), val);
  }

  void generate_overflow_branch(const spirv::Value &cond_v,
                                const std::string &op,
                                const std::string &tb) {
    spirv::Value cond =
        ir_->ne(cond_v, ir_->cast(cond_v.stype, ir_->const_i32_zero_));
    spirv::Label then_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();
    ir_->make_inst(spv::OpSelectionMerge, merge_label,
                   spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
    // then block
    ir_->start_label(then_label);
    ir_->call_debugprintf(op + " overflow detected in " + tb, {});
    ir_->make_inst(spv::OpBranch, merge_label);
    // merge label
    ir_->start_label(merge_label);
  }

  spirv::Value generate_uadd_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    std::vector<std::tuple<spirv::SType, std::string, size_t>>
        struct_components_;
    struct_components_.emplace_back(a.stype, "result", 0);
    struct_components_.emplace_back(a.stype, "carry",
                                    ir_->get_primitive_type_size(a.stype.dt));
    auto struct_type = ir_->create_struct_type(struct_components_);
    auto add_carry = ir_->make_value(spv::OpIAddCarry, struct_type, a, b);
    auto result =
        ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 0);
    auto carry =
        ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 1);
    generate_overflow_branch(carry, "Addition", tb);
    return result;
  }

  spirv::Value generate_usub_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    std::vector<std::tuple<spirv::SType, std::string, size_t>>
        struct_components_;
    struct_components_.emplace_back(a.stype, "result", 0);
    struct_components_.emplace_back(a.stype, "borrow",
                                    ir_->get_primitive_type_size(a.stype.dt));
    auto struct_type = ir_->create_struct_type(struct_components_);
    auto add_carry = ir_->make_value(spv::OpISubBorrow, struct_type, a, b);
    auto result =
        ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 0);
    auto borrow =
        ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 1);
    generate_overflow_branch(borrow, "Subtraction", tb);
    return result;
  }

  spirv::Value generate_sadd_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow iff (sign(a) == sign(b)) && (sign(a) != sign(result))
    auto result = ir_->make_value(spv::OpIAdd, a.stype, a, b);
    auto zero = ir_->int_immediate_number(a.stype, 0);
    auto a_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), a, zero);
    auto b_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), b, zero);
    auto r_sign =
        ir_->make_value(spv::OpSLessThan, ir_->bool_type(), result, zero);
    auto a_eq_b =
        ir_->make_value(spv::OpLogicalEqual, ir_->bool_type(), a_sign, b_sign);
    auto a_neq_r = ir_->make_value(spv::OpLogicalNotEqual, ir_->bool_type(),
                                   a_sign, r_sign);
    auto overflow =
        ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(), a_eq_b, a_neq_r);
    generate_overflow_branch(overflow, "Addition", tb);
    return result;
  }

  spirv::Value generate_ssub_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow iff (sign(a) != sign(b)) && (sign(a) != sign(result))
    auto result = ir_->make_value(spv::OpISub, a.stype, a, b);
    auto zero = ir_->int_immediate_number(a.stype, 0);
    auto a_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), a, zero);
    auto b_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), b, zero);
    auto r_sign =
        ir_->make_value(spv::OpSLessThan, ir_->bool_type(), result, zero);
    auto a_neq_b = ir_->make_value(spv::OpLogicalNotEqual, ir_->bool_type(),
                                   a_sign, b_sign);
    auto a_neq_r = ir_->make_value(spv::OpLogicalNotEqual, ir_->bool_type(),
                                   a_sign, r_sign);
    auto overflow =
        ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(), a_neq_b, a_neq_r);
    generate_overflow_branch(overflow, "Subtraction", tb);
    return result;
  }

  spirv::Value generate_umul_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow iff high bits != 0
    std::vector<std::tuple<spirv::SType, std::string, size_t>>
        struct_components_;
    struct_components_.emplace_back(a.stype, "low", 0);
    struct_components_.emplace_back(a.stype, "high",
                                    ir_->get_primitive_type_size(a.stype.dt));
    auto struct_type = ir_->create_struct_type(struct_components_);
    auto mul_ext = ir_->make_value(spv::OpUMulExtended, struct_type, a, b);
    auto low = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 0);
    auto high = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 1);
    generate_overflow_branch(high, "Multiplication", tb);
    return low;
  }

  spirv::Value generate_smul_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow if high bits are not all sign bit (0 if positive, -1 if
    // negative) or the sign bit of the low bits is not the expected sign bit.
    std::vector<std::tuple<spirv::SType, std::string, size_t>>
        struct_components_;
    struct_components_.emplace_back(a.stype, "low", 0);
    struct_components_.emplace_back(a.stype, "high",
                                    ir_->get_primitive_type_size(a.stype.dt));
    auto struct_type = ir_->create_struct_type(struct_components_);
    auto mul_ext = ir_->make_value(spv::OpSMulExtended, struct_type, a, b);
    auto low = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 0);
    auto high = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 1);
    auto zero = ir_->int_immediate_number(a.stype, 0);
    auto minus_one = ir_->int_immediate_number(a.stype, -1);
    auto a_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), a, zero);
    auto b_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), b, zero);
    auto a_not_zero = ir_->ne(a, zero);
    auto b_not_zero = ir_->ne(b, zero);
    auto a_b_not_zero = ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(),
                                        a_not_zero, b_not_zero);
    auto low_sign =
        ir_->make_value(spv::OpSLessThan, ir_->bool_type(), low, zero);
    auto expected_sign = ir_->make_value(spv::OpLogicalNotEqual,
                                         ir_->bool_type(), a_sign, b_sign);
    expected_sign = ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(),
                                    expected_sign, a_b_not_zero);
    auto not_expected_sign = ir_->ne(low_sign, expected_sign);
    auto expected_high = ir_->select(expected_sign, minus_one, zero);
    auto not_expected_high = ir_->ne(high, expected_high);
    auto overflow = ir_->make_value(spv::OpLogicalOr, ir_->bool_type(),
                                    not_expected_high, not_expected_sign);
    generate_overflow_branch(overflow, "Multiplication", tb);
    return low;
  }

  spirv::Value generate_ushl_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow iff a << b >> b != a
    auto result = ir_->make_value(spv::OpShiftLeftLogical, a.stype, a, b);
    auto restore =
        ir_->make_value(spv::OpShiftRightLogical, a.stype, result, b);
    auto overflow = ir_->ne(a, restore);
    generate_overflow_branch(overflow, "Shift left", tb);
    return result;
  }

  spirv::Value generate_sshl_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow iff a << b >> b != a
    auto result = ir_->make_value(spv::OpShiftLeftLogical, a.stype, a, b);
    auto restore =
        ir_->make_value(spv::OpShiftRightArithmetic, a.stype, result, b);
    auto overflow = ir_->ne(a, restore);
    generate_overflow_branch(overflow, "Shift left", tb);
    return result;
  }

  void visit(BinaryOpStmt *bin) override {
    const auto lhs_name = bin->lhs->raw_name();
    const auto rhs_name = bin->rhs->raw_name();
    const auto bin_name = bin->raw_name();
    const auto op_type = bin->op_type;

    spirv::SType dst_type = ir_->get_primitive_type(bin->element_type());
    spirv::Value lhs_value = ir_->query_value(lhs_name);
    spirv::Value rhs_value = ir_->query_value(rhs_name);
    spirv::Value bin_value = spirv::Value();

    TI_WARN_IF(lhs_value.stype.id != rhs_value.stype.id,
               "${} type {} != ${} type {}\n{}", lhs_name,
               lhs_value.stype.dt->to_string(), rhs_name,
               rhs_value.stype.dt->to_string(), bin->tb);

    bool debug = caps_->get(DeviceCapability::spirv_has_non_semantic_info);

    if (debug && op_type == BinaryOpType::add && is_integral(dst_type.dt)) {
      if (is_unsigned(dst_type.dt)) {
        bin_value = generate_uadd_overflow(lhs_value, rhs_value, bin->tb);
      } else {
        bin_value = generate_sadd_overflow(lhs_value, rhs_value, bin->tb);
      }
      bin_value = ir_->cast(dst_type, bin_value);
    } else if (debug && op_type == BinaryOpType::sub &&
               is_integral(dst_type.dt)) {
      if (is_unsigned(dst_type.dt)) {
        bin_value = generate_usub_overflow(lhs_value, rhs_value, bin->tb);
      } else {
        bin_value = generate_ssub_overflow(lhs_value, rhs_value, bin->tb);
      }
      bin_value = ir_->cast(dst_type, bin_value);
    } else if (debug && op_type == BinaryOpType::mul &&
               is_integral(dst_type.dt)) {
      if (is_unsigned(dst_type.dt)) {
        bin_value = generate_umul_overflow(lhs_value, rhs_value, bin->tb);
      } else {
        bin_value = generate_smul_overflow(lhs_value, rhs_value, bin->tb);
      }
      bin_value = ir_->cast(dst_type, bin_value);
    }
#define BINARY_OP_TO_SPIRV_ARTHIMATIC(op, func)  \
  else if (op_type == BinaryOpType::op) {        \
    bin_value = ir_->func(lhs_value, rhs_value); \
    bin_value = ir_->cast(dst_type, bin_value);  \
  }

    BINARY_OP_TO_SPIRV_ARTHIMATIC(add, add)
    BINARY_OP_TO_SPIRV_ARTHIMATIC(sub, sub)
    BINARY_OP_TO_SPIRV_ARTHIMATIC(mul, mul)
    BINARY_OP_TO_SPIRV_ARTHIMATIC(div, div)
    BINARY_OP_TO_SPIRV_ARTHIMATIC(mod, mod)
#undef BINARY_OP_TO_SPIRV_ARTHIMATIC

#define BINARY_OP_TO_SPIRV_BITWISE(op, sym)                                \
  else if (op_type == BinaryOpType::op) {                                  \
    bin_value = ir_->make_value(spv::sym, dst_type, lhs_value, rhs_value); \
  }

    else if (debug && op_type == BinaryOpType::bit_shl) {
      if (is_unsigned(dst_type.dt)) {
        bin_value = generate_ushl_overflow(lhs_value, rhs_value, bin->tb);
      } else {
        bin_value = generate_sshl_overflow(lhs_value, rhs_value, bin->tb);
      }
    }
    BINARY_OP_TO_SPIRV_BITWISE(bit_and, OpBitwiseAnd)
    BINARY_OP_TO_SPIRV_BITWISE(bit_or, OpBitwiseOr)
    BINARY_OP_TO_SPIRV_BITWISE(bit_xor, OpBitwiseXor)
    BINARY_OP_TO_SPIRV_BITWISE(bit_shl, OpShiftLeftLogical)
    // NOTE: `OpShiftRightArithmetic` will treat the first bit as sign bit even
    // it's the unsigned type
    else if (op_type == BinaryOpType::bit_sar) {
      bin_value = ir_->make_value(is_unsigned(dst_type.dt)
                                      ? spv::OpShiftRightLogical
                                      : spv::OpShiftRightArithmetic,
                                  dst_type, lhs_value, rhs_value);
    }
#undef BINARY_OP_TO_SPIRV_BITWISE

#define BINARY_OP_TO_SPIRV_LOGICAL(op, func)     \
  else if (op_type == BinaryOpType::op) {        \
    bin_value = ir_->func(lhs_value, rhs_value); \
    bin_value = ir_->cast(dst_type, bin_value);  \
  }

    BINARY_OP_TO_SPIRV_LOGICAL(cmp_lt, lt)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_le, le)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_gt, gt)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_ge, ge)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_eq, eq)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_ne, ne)
#undef BINARY_OP_TO_SPIRV_LOGICAL

#define FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC(op, instruction, instruction_id,   \
                                            max_bits)                          \
  else if (op_type == BinaryOpType::op) {                                      \
    const uint32_t instruction = instruction_id;                               \
    if (is_real(bin->element_type())) {                                        \
      if (data_type_bits(bin->element_type()) > max_bits) {                    \
        TI_ERROR(                                                              \
            "[glsl450] the operand type of instruction {}({}) must <= {}bits", \
            #instruction, instruction_id, max_bits);                           \
      }                                                                        \
      bin_value =                                                              \
          ir_->call_glsl450(dst_type, instruction, lhs_value, rhs_value);      \
    } else {                                                                   \
      TI_NOT_IMPLEMENTED                                                       \
    }                                                                          \
  }

    FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC(atan2, Atan2, 25, 32)
    FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC(pow, Pow, 26, 32)
#undef FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC

#define BINARY_OP_TO_SPIRV_FUNC(op, S_inst, S_inst_id, U_inst, U_inst_id,      \
                                F_inst, F_inst_id)                             \
  else if (op_type == BinaryOpType::op) {                                      \
    const uint32_t S_inst = S_inst_id;                                         \
    const uint32_t U_inst = U_inst_id;                                         \
    const uint32_t F_inst = F_inst_id;                                         \
    const auto dst_dt = bin->element_type();                                   \
    if (is_integral(dst_dt)) {                                                 \
      if (is_signed(dst_dt)) {                                                 \
        bin_value = ir_->call_glsl450(dst_type, S_inst, lhs_value, rhs_value); \
      } else {                                                                 \
        bin_value = ir_->call_glsl450(dst_type, U_inst, lhs_value, rhs_value); \
      }                                                                        \
    } else if (is_real(dst_dt)) {                                              \
      bin_value = ir_->call_glsl450(dst_type, F_inst, lhs_value, rhs_value);   \
    } else {                                                                   \
      TI_NOT_IMPLEMENTED                                                       \
    }                                                                          \
  }

    BINARY_OP_TO_SPIRV_FUNC(min, SMin, 39, UMin, 38, FMin, 37)
    BINARY_OP_TO_SPIRV_FUNC(max, SMax, 42, UMax, 41, FMax, 40)
#undef BINARY_OP_TO_SPIRV_FUNC
    else if (op_type == BinaryOpType::truediv) {
      lhs_value = ir_->cast(dst_type, lhs_value);
      rhs_value = ir_->cast(dst_type, rhs_value);
      bin_value = ir_->div(lhs_value, rhs_value);
    }
    else {TI_NOT_IMPLEMENTED} ir_->register_value(bin_name, bin_value);
  }

  void visit(TernaryOpStmt *tri) override {
    TI_ASSERT(tri->op_type == TernaryOpType::select);
    spirv::Value op1 = ir_->query_value(tri->op1->raw_name());
    spirv::Value op2 = ir_->query_value(tri->op2->raw_name());
    spirv::Value op3 = ir_->query_value(tri->op3->raw_name());
    spirv::Value tri_val =
        ir_->cast(ir_->get_primitive_type(tri->element_type()),
                  ir_->select(ir_->cast(ir_->bool_type(), op1), op2, op3));
    ir_->register_value(tri->raw_name(), tri_val);
  }

  inline bool ends_with(std::string const &value, std::string const &ending) {
    if (ending.size() > value.size())
      return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
  }

  void visit(TexturePtrStmt *stmt) override {
    spirv::Value val;

    int arg_id = stmt->arg_load_stmt->as<ArgLoadStmt>()->arg_id;
    if (argid_to_tex_value_.find(arg_id) != argid_to_tex_value_.end()) {
      val = argid_to_tex_value_.at(arg_id);
    } else {
      if (stmt->is_storage) {
        BufferFormat format = stmt->format;

        int binding = binding_head_++;
        val =
            ir_->storage_image_argument(/*num_channels=*/4, stmt->dimensions,
                                        /*descriptor_set=*/0, binding, format);
        TextureBind bind;
        bind.arg_id = arg_id;
        bind.binding = binding;
        bind.is_storage = true;
        texture_binds_.push_back(bind);
        argid_to_tex_value_[arg_id] = val;
      } else {
        int binding = binding_head_++;
        val = ir_->texture_argument(/*num_channels=*/4, stmt->dimensions,
                                    /*descriptor_set=*/0, binding);
        TextureBind bind;
        bind.arg_id = arg_id;
        bind.binding = binding;
        texture_binds_.push_back(bind);
        argid_to_tex_value_[arg_id] = val;
      }
    }

    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(TextureOpStmt *stmt) override {
    spirv::Value tex = ir_->query_value(stmt->texture_ptr->raw_name());
    spirv::Value val;
    if (stmt->op == TextureOpType::kSampleLod ||
        stmt->op == TextureOpType::kFetchTexel) {
      // Texture Ops
      std::vector<spirv::Value> args;
      for (int i = 0; i < stmt->args.size() - 1; i++) {
        args.push_back(ir_->query_value(stmt->args[i]->raw_name()));
      }
      spirv::Value lod = ir_->query_value(stmt->args.back()->raw_name());
      if (stmt->op == TextureOpType::kSampleLod) {
        // Sample
        val = ir_->sample_texture(tex, args, lod);
      } else if (stmt->op == TextureOpType::kFetchTexel) {
        // Texel fetch
        val = ir_->fetch_texel(tex, args, lod);
      }
      ir_->register_value(stmt->raw_name(), val);
    } else if (stmt->op == TextureOpType::kLoad ||
               stmt->op == TextureOpType::kStore) {
      // Image Ops
      std::vector<spirv::Value> args;
      for (int i = 0; i < stmt->args.size(); i++) {
        args.push_back(ir_->query_value(stmt->args[i]->raw_name()));
      }
      if (stmt->op == TextureOpType::kLoad) {
        // Image Load
        val = ir_->image_load(tex, args);
        ir_->register_value(stmt->raw_name(), val);
      } else if (stmt->op == TextureOpType::kStore) {
        // Image Store
        ir_->image_store(tex, args);
      }
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(InternalFuncStmt *stmt) override {
    spirv::Value val;

    if (stmt->func_name == "composite_extract_0") {
      val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(),
                            ir_->query_value(stmt->args[0]->raw_name()), 0);
    } else if (stmt->func_name == "composite_extract_1") {
      val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(),
                            ir_->query_value(stmt->args[0]->raw_name()), 1);
    } else if (stmt->func_name == "composite_extract_2") {
      val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(),
                            ir_->query_value(stmt->args[0]->raw_name()), 2);
    } else if (stmt->func_name == "composite_extract_3") {
      val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(),
                            ir_->query_value(stmt->args[0]->raw_name()), 3);
    }

    const std::unordered_set<std::string> reduction_ops{
        "subgroupAdd", "subgroupMul", "subgroupMin", "subgroupMax",
        "subgroupAnd", "subgroupOr",  "subgroupXor"};

    const std::unordered_set<std::string> inclusive_scan_ops{
        "subgroupInclusiveAdd", "subgroupInclusiveMul", "subgroupInclusiveMin",
        "subgroupInclusiveMax", "subgroupInclusiveAnd", "subgroupInclusiveOr",
        "subgroupInclusiveXor"};

    const std::unordered_set<std::string> shuffle_ops{
        "subgroupShuffleDown", "subgroupShuffleUp", "subgroupShuffle"};

    if (stmt->func_name == "workgroupBarrier") {
      ir_->make_inst(
          spv::OpControlBarrier,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeWorkgroup),
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeWorkgroup),
          ir_->int_immediate_number(
              ir_->i32_type(), spv::MemorySemanticsWorkgroupMemoryMask |
                                   spv::MemorySemanticsAcquireReleaseMask));
      val = ir_->const_i32_zero_;
    } else if (stmt->func_name == "localInvocationId") {
      val = ir_->cast(ir_->i32_type(), ir_->get_local_invocation_id(0));
    } else if (stmt->func_name == "globalInvocationId") {
      val = ir_->cast(ir_->i32_type(), ir_->get_global_invocation_id(0));
    } else if (stmt->func_name == "workgroupMemoryBarrier") {
      ir_->make_inst(
          spv::OpMemoryBarrier,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeWorkgroup),
          ir_->int_immediate_number(
              ir_->i32_type(), spv::MemorySemanticsWorkgroupMemoryMask |
                                   spv::MemorySemanticsAcquireReleaseMask));
      val = ir_->const_i32_zero_;
    } else if (stmt->func_name == "subgroupElect") {
      val = ir_->make_value(
          spv::OpGroupNonUniformElect, ir_->bool_type(),
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup));
      val = ir_->cast(ir_->i32_type(), val);
    } else if (stmt->func_name == "subgroupBarrier") {
      ir_->make_inst(
          spv::OpControlBarrier,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup),
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup),
          ir_->const_i32_zero_);
      val = ir_->const_i32_zero_;
    } else if (stmt->func_name == "subgroupMemoryBarrier") {
      ir_->make_inst(
          spv::OpMemoryBarrier,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup),
          ir_->const_i32_zero_);
      val = ir_->const_i32_zero_;
    } else if (stmt->func_name == "subgroupSize") {
      val = ir_->cast(ir_->i32_type(), ir_->get_subgroup_size());
    } else if (stmt->func_name == "subgroupInvocationId") {
      val = ir_->cast(ir_->i32_type(), ir_->get_subgroup_invocation_id());
    } else if (stmt->func_name == "subgroupBroadcast") {
      auto value = ir_->query_value(stmt->args[0]->raw_name());
      auto index = ir_->query_value(stmt->args[1]->raw_name());
      val = ir_->make_value(
          spv::OpGroupNonUniformBroadcast, value.stype,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup), value,
          index);
    } else if (reduction_ops.find(stmt->func_name) != reduction_ops.end() ||
               inclusive_scan_ops.find(stmt->func_name) !=
                   inclusive_scan_ops.end()) {
      auto arg = ir_->query_value(stmt->args[0]->raw_name());
      auto stype = ir_->get_primitive_type(stmt->args[0]->ret_type);
      spv::Op spv_op;

      if (ends_with(stmt->func_name, "Add")) {
        if (is_integral(stmt->args[0]->ret_type)) {
          spv_op = spv::OpGroupNonUniformIAdd;
        } else {
          spv_op = spv::OpGroupNonUniformFAdd;
        }
      } else if (ends_with(stmt->func_name, "Mul")) {
        if (is_integral(stmt->args[0]->ret_type)) {
          spv_op = spv::OpGroupNonUniformIMul;
        } else {
          spv_op = spv::OpGroupNonUniformFMul;
        }
      } else if (ends_with(stmt->func_name, "Min")) {
        if (is_integral(stmt->args[0]->ret_type)) {
          if (is_signed(stmt->args[0]->ret_type)) {
            spv_op = spv::OpGroupNonUniformSMin;
          } else {
            spv_op = spv::OpGroupNonUniformUMin;
          }
        } else {
          spv_op = spv::OpGroupNonUniformFMin;
        }
      } else if (ends_with(stmt->func_name, "Max")) {
        if (is_integral(stmt->args[0]->ret_type)) {
          if (is_signed(stmt->args[0]->ret_type)) {
            spv_op = spv::OpGroupNonUniformSMax;
          } else {
            spv_op = spv::OpGroupNonUniformUMax;
          }
        } else {
          spv_op = spv::OpGroupNonUniformFMax;
        }
      } else if (ends_with(stmt->func_name, "And")) {
        spv_op = spv::OpGroupNonUniformBitwiseAnd;
      } else if (ends_with(stmt->func_name, "Or")) {
        spv_op = spv::OpGroupNonUniformBitwiseOr;
      } else if (ends_with(stmt->func_name, "Xor")) {
        spv_op = spv::OpGroupNonUniformBitwiseXor;
      } else {
        TI_ERROR("Unsupported operation: {}", stmt->func_name);
      }

      spv::GroupOperation group_op;

      if (reduction_ops.find(stmt->func_name) != reduction_ops.end()) {
        group_op = spv::GroupOperationReduce;
      } else if (inclusive_scan_ops.find(stmt->func_name) !=
                 inclusive_scan_ops.end()) {
        group_op = spv::GroupOperationInclusiveScan;
      }

      val = ir_->make_value(
          spv_op, stype,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup),
          group_op, arg);
    } else if (shuffle_ops.find(stmt->func_name) != shuffle_ops.end()) {
      auto arg0 = ir_->query_value(stmt->args[0]->raw_name());
      auto arg1 = ir_->query_value(stmt->args[1]->raw_name());
      auto stype = ir_->get_primitive_type(stmt->args[0]->ret_type);
      spv::Op spv_op;

      if (ends_with(stmt->func_name, "Down")) {
        spv_op = spv::OpGroupNonUniformShuffleDown;
      } else if (ends_with(stmt->func_name, "Up")) {
        spv_op = spv::OpGroupNonUniformShuffleUp;
      } else if (ends_with(stmt->func_name, "Shuffle")) {
        spv_op = spv::OpGroupNonUniformShuffle;
      } else {
        TI_ERROR("Unsupported operation: {}", stmt->func_name);
      }

      val = ir_->make_value(
          spv_op, stype,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup), arg0,
          arg1);
    }
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(AtomicOpStmt *stmt) override {
    const auto dt = stmt->dest->element_type().ptr_removed();

    spirv::Value data = ir_->query_value(stmt->val->raw_name());
    spirv::Value val;
    bool use_subgroup_reduction = false;

    if (stmt->is_reduction &&
        caps_->get(DeviceCapability::spirv_has_subgroup_arithmetic)) {
      spv::Op atomic_op = spv::OpNop;
      bool negation = false;
      if (is_integral(dt)) {
        if (stmt->op_type == AtomicOpType::add) {
          atomic_op = spv::OpGroupIAdd;
        } else if (stmt->op_type == AtomicOpType::sub) {
          atomic_op = spv::OpGroupIAdd;
          negation = true;
        } else if (stmt->op_type == AtomicOpType::min) {
          atomic_op = is_signed(dt) ? spv::OpGroupSMin : spv::OpGroupUMin;
        } else if (stmt->op_type == AtomicOpType::max) {
          atomic_op = is_signed(dt) ? spv::OpGroupSMax : spv::OpGroupUMax;
        }
      } else if (is_real(dt)) {
        if (stmt->op_type == AtomicOpType::add) {
          atomic_op = spv::OpGroupFAdd;
        } else if (stmt->op_type == AtomicOpType::sub) {
          atomic_op = spv::OpGroupFAdd;
          negation = true;
        } else if (stmt->op_type == AtomicOpType::min) {
          atomic_op = spv::OpGroupFMin;
        } else if (stmt->op_type == AtomicOpType::max) {
          atomic_op = spv::OpGroupFMax;
        }
      }

      if (atomic_op != spv::OpNop) {
        spirv::Value scope_subgroup =
            ir_->int_immediate_number(ir_->i32_type(), 3);
        spirv::Value operation_reduce = ir_->const_i32_zero_;
        if (negation) {
          if (is_integral(dt)) {
            data = ir_->make_value(spv::OpSNegate, data.stype, data);
          } else {
            data = ir_->make_value(spv::OpFNegate, data.stype, data);
          }
        }
        data = ir_->make_value(atomic_op, ir_->get_primitive_type(dt),
                               scope_subgroup, operation_reduce, data);
        val = data;
        use_subgroup_reduction = true;
      }
    }

    spirv::Label then_label;
    spirv::Label merge_label;

    if (use_subgroup_reduction) {
      spirv::Value subgroup_id = ir_->get_subgroup_invocation_id();
      spirv::Value cond = ir_->make_value(spv::OpIEqual, ir_->bool_type(),
                                          subgroup_id, ir_->const_i32_zero_);

      then_label = ir_->new_label();
      merge_label = ir_->new_label();
      ir_->make_inst(spv::OpSelectionMerge, merge_label,
                     spv::SelectionControlMaskNone);
      ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
      ir_->start_label(then_label);
    }

    spirv::Value addr_ptr;

    if (dt->is_primitive(PrimitiveTypeID::f64)) {
      if (caps_->get(DeviceCapability::spirv_has_atomic_float64_add) &&
          stmt->op_type == AtomicOpType::add) {
        addr_ptr = at_buffer(stmt->dest, dt);
      } else {
        addr_ptr = at_buffer(stmt->dest, ir_->get_taichi_uint_type(dt));
      }
    } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
      if (caps_->get(DeviceCapability::spirv_has_atomic_float_add) &&
          stmt->op_type == AtomicOpType::add) {
        addr_ptr = at_buffer(stmt->dest, dt);
      } else {
        addr_ptr = at_buffer(stmt->dest, ir_->get_taichi_uint_type(dt));
      }
    } else {
      if (stmt->dest->is<MatrixPtrStmt>()) {
        // Shared arrays have already created an accesschain, use it directly.
        addr_ptr = ir_->query_value(stmt->dest->raw_name());
      } else {
        addr_ptr = at_buffer(stmt->dest, dt);
      }
    }

    auto ret_type = ir_->get_primitive_type(dt);

    if (is_real(dt)) {
      spv::Op atomic_fp_op;
      if (stmt->op_type == AtomicOpType::add) {
        atomic_fp_op = spv::OpAtomicFAddEXT;
      }

      bool use_native_atomics = false;

      if (dt->is_primitive(PrimitiveTypeID::f64)) {
        if (caps_->get(DeviceCapability::spirv_has_atomic_float64_add) &&
            stmt->op_type == AtomicOpType::add) {
          use_native_atomics = true;
        }
      } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
        if (caps_->get(DeviceCapability::spirv_has_atomic_float_add) &&
            stmt->op_type == AtomicOpType::add) {
          use_native_atomics = true;
        }
      } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
        if (caps_->get(DeviceCapability::spirv_has_atomic_float16_add) &&
            stmt->op_type == AtomicOpType::add) {
          use_native_atomics = true;
        }
      }

      if (use_native_atomics) {
        val =
            ir_->make_value(atomic_fp_op, ir_->get_primitive_type(dt), addr_ptr,
                            /*scope=*/ir_->const_i32_one_,
                            /*semantics=*/ir_->const_i32_zero_, data);
      } else {
        val = ir_->float_atomic(stmt->op_type, addr_ptr, data, dt);
      }
    } else if (is_integral(dt)) {
      bool use_native_atomics = false;
      spv::Op op;
      if (stmt->op_type == AtomicOpType::add) {
        op = spv::OpAtomicIAdd;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::sub) {
        op = spv::OpAtomicISub;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::mul) {
        addr_ptr = at_buffer(stmt->dest, ir_->get_taichi_uint_type(dt));
        val = ir_->integer_atomic(stmt->op_type, addr_ptr, data, dt);
        use_native_atomics = false;
      } else if (stmt->op_type == AtomicOpType::min) {
        op = is_signed(dt) ? spv::OpAtomicSMin : spv::OpAtomicUMin;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::max) {
        op = is_signed(dt) ? spv::OpAtomicSMax : spv::OpAtomicUMax;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::bit_or) {
        op = spv::OpAtomicOr;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::bit_and) {
        op = spv::OpAtomicAnd;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::bit_xor) {
        op = spv::OpAtomicXor;
        use_native_atomics = true;
      } else {
        TI_NOT_IMPLEMENTED
      }

      if (use_native_atomics) {
        auto uint_type = ir_->get_primitive_uint_type(dt);

        if (data.stype.id != addr_ptr.stype.element_type_id) {
          data = ir_->make_value(spv::OpBitcast, ret_type, data);
        }

        // Semantics = (UniformMemory 0x40) | (AcquireRelease 0x8)
        ir_->make_inst(
            spv::OpMemoryBarrier, ir_->const_i32_one_,
            ir_->uint_immediate_number(
                ir_->u32_type(), spv::MemorySemanticsAcquireReleaseMask |
                                     spv::MemorySemanticsUniformMemoryMask));
        val = ir_->make_value(op, ret_type, addr_ptr,
                              /*scope=*/ir_->const_i32_one_,
                              /*semantics=*/ir_->const_i32_zero_, data);

        if (val.stype.id != ret_type.id) {
          val = ir_->make_value(spv::OpBitcast, ret_type, val);
        }
      }
    } else {
      TI_NOT_IMPLEMENTED
    }

    if (use_subgroup_reduction) {
      ir_->make_inst(spv::OpBranch, merge_label);
      ir_->start_label(merge_label);
    }

    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(IfStmt *if_stmt) override {
    spirv::Value cond_v = ir_->query_value(if_stmt->cond->raw_name());
    spirv::Value cond =
        ir_->ne(cond_v, ir_->cast(cond_v.stype, ir_->const_i32_zero_));
    spirv::Label then_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();
    spirv::Label else_label = ir_->new_label();
    ir_->make_inst(spv::OpSelectionMerge, merge_label,
                   spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, cond, then_label, else_label);
    // then block
    ir_->start_label(then_label);
    if (if_stmt->true_statements) {
      if_stmt->true_statements->accept(this);
    }
    // ContinueStmt must be in IfStmt
    if (gen_label_) {  // Skip OpBranch, because ContinueStmt already generated
                       // one
      gen_label_ = false;
    } else {
      ir_->make_inst(spv::OpBranch, merge_label);
    }
    // else block
    ir_->start_label(else_label);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
    if (gen_label_) {
      gen_label_ = false;
    } else {
      ir_->make_inst(spv::OpBranch, merge_label);
    }
    // merge label
    ir_->start_label(merge_label);
  }

  void visit(RangeForStmt *for_stmt) override {
    auto loop_var_name = for_stmt->raw_name();
    // Must get init label after making value(to make sure they are correct)
    spirv::Label init_label = ir_->current_label();
    spirv::Label head_label = ir_->new_label();
    spirv::Label body_label = ir_->new_label();
    spirv::Label continue_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();

    spirv::Value begin_ = ir_->query_value(for_stmt->begin->raw_name());
    spirv::Value end_ = ir_->query_value(for_stmt->end->raw_name());
    spirv::Value init_value;
    spirv::Value extent_value;
    if (!for_stmt->reversed) {
      init_value = begin_;
      extent_value = end_;
    } else {
      // reversed for loop
      init_value = ir_->sub(end_, ir_->const_i32_one_);
      extent_value = begin_;
    }
    ir_->make_inst(spv::OpBranch, head_label);

    // Loop head
    ir_->start_label(head_label);
    spirv::PhiValue loop_var = ir_->make_phi(init_value.stype, 2);
    loop_var.set_incoming(0, init_value, init_label);
    spirv::Value loop_cond;
    if (!for_stmt->reversed) {
      loop_cond = ir_->lt(loop_var, extent_value);
    } else {
      loop_cond = ir_->ge(loop_var, extent_value);
    }
    ir_->make_inst(spv::OpLoopMerge, merge_label, continue_label,
                   spv::LoopControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, loop_cond, body_label,
                   merge_label);

    // loop body
    ir_->start_label(body_label);
    push_loop_control_labels(continue_label, merge_label);
    ir_->register_value(loop_var_name, spirv::Value(loop_var));
    for_stmt->body->accept(this);
    pop_loop_control_labels();
    ir_->make_inst(spv::OpBranch, continue_label);

    // loop continue
    ir_->start_label(continue_label);
    spirv::Value next_value;
    if (!for_stmt->reversed) {
      next_value = ir_->add(loop_var, ir_->const_i32_one_);
    } else {
      next_value = ir_->sub(loop_var, ir_->const_i32_one_);
    }
    loop_var.set_incoming(1, next_value, ir_->current_label());
    ir_->make_inst(spv::OpBranch, head_label);
    // loop merge
    ir_->start_label(merge_label);
  }

  void visit(WhileStmt *stmt) override {
    spirv::Label head_label = ir_->new_label();
    spirv::Label body_label = ir_->new_label();
    spirv::Label continue_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();
    ir_->make_inst(spv::OpBranch, head_label);

    // Loop head
    ir_->start_label(head_label);
    ir_->make_inst(spv::OpLoopMerge, merge_label, continue_label,
                   spv::LoopControlMaskNone);
    ir_->make_inst(spv::OpBranch, body_label);

    // loop body
    ir_->start_label(body_label);
    push_loop_control_labels(continue_label, merge_label);
    stmt->body->accept(this);
    pop_loop_control_labels();
    ir_->make_inst(spv::OpBranch, continue_label);

    // loop continue
    ir_->start_label(continue_label);
    ir_->make_inst(spv::OpBranch, head_label);

    // loop merge
    ir_->start_label(merge_label);
  }

  void visit(WhileControlStmt *stmt) override {
    spirv::Value cond_v = ir_->query_value(stmt->cond->raw_name());
    spirv::Value cond =
        ir_->eq(cond_v, ir_->cast(cond_v.stype, ir_->const_i32_zero_));
    spirv::Label then_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();

    ir_->make_inst(spv::OpSelectionMerge, merge_label,
                   spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
    ir_->start_label(then_label);
    ir_->make_inst(spv::OpBranch, current_merge_label());  // break;
    ir_->start_label(merge_label);
  }

  void visit(ContinueStmt *stmt) override {
    auto stmt_in_off_for = [stmt]() {
      TI_ASSERT(stmt->scope != nullptr);
      if (auto *offl = stmt->scope->cast<OffloadedStmt>(); offl) {
        TI_ASSERT(offl->task_type == OffloadedStmt::TaskType::range_for ||
                  offl->task_type == OffloadedStmt::TaskType::struct_for);
        return true;
      }
      return false;
    };
    if (stmt_in_off_for()) {
      // Return means end THIS main loop and start next loop, not exit kernel
      ir_->make_inst(spv::OpBranch, return_label());
    } else {
      ir_->make_inst(spv::OpBranch, current_continue_label());
    }
    gen_label_ = true;  // Only ContinueStmt will cause duplicate OpBranch,
                        // which should be eliminated
  }

 private:
  void emit_headers() {
    /*
    for (int root = 0; root < compiled_structs_.size(); ++root) {
      get_buffer_value({BufferType::Root, root});
    }
    */
    std::array<int, 3> group_size = {
        task_attribs_.advisory_num_threads_per_group, 1, 1};
    ir_->set_work_group_size(group_size);
    std::vector<spirv::Value> buffers;
    if (caps_->get(DeviceCapability::spirv_version) > 0x10300) {
      buffers = shared_array_binds_;
      // One buffer can be bound to different bind points but has to be unique
      // in OpEntryPoint interface declarations.
      // From Spec: before SPIR-V version 1.4, duplication of these interface id
      // is tolerated. Starting with version 1.4, an interface id must not
      // appear more than once.
      std::unordered_set<spirv::Value, spirv::ValueHasher> entry_point_values;
      for (const auto &bb : task_attribs_.buffer_binds) {
        for (auto &it : buffer_value_map_) {
          if (it.first.first == bb.buffer) {
            entry_point_values.insert(it.second);
          }
        }
      }
      buffers.insert(buffers.end(), entry_point_values.begin(),
                     entry_point_values.end());
    }
    ir_->commit_kernel_function(kernel_function_, "main", buffers,
                                group_size);  // kernel entry
  }

  void generate_serial_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::serial;
    task_attribs_.advisory_total_num_threads = 1;
    task_attribs_.advisory_num_threads_per_group = 1;

    // The computation for a single work is wrapped inside a function, so that
    // we can do grid-strided loop.
    ir_->start_function(kernel_function_);
    spirv::Value cond =
        ir_->eq(ir_->get_global_invocation_id(0),
                ir_->uint_immediate_number(
                    ir_->u32_type(), 0));  // if (gl_GlobalInvocationID.x > 0)
    spirv::Label then_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();
    kernel_return_label_ = merge_label;

    ir_->make_inst(spv::OpSelectionMerge, merge_label,
                   spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
    ir_->start_label(then_label);

    // serial kernel
    stmt->body->accept(this);

    ir_->make_inst(spv::OpBranch, merge_label);
    ir_->start_label(merge_label);
    ir_->make_inst(spv::OpReturn);       // return;
    ir_->make_inst(spv::OpFunctionEnd);  // } Close kernel

    task_attribs_.buffer_binds = get_buffer_binds();
    task_attribs_.texture_binds = get_texture_binds();
  }

  void gen_array_range(Stmt *stmt) {
    /* Fix issue 7493
     *
     * Prevent repeated range generation for the same array
     * when loop range has multiple dimensions.
     */
    if (ir_->check_value_existence(stmt->raw_name())) {
      return;
    }
    int num_operands = stmt->num_operands();
    for (int i = 0; i < num_operands; i++) {
      gen_array_range(stmt->operand(i));
    }
    offload_loop_motion_.insert(stmt);
    stmt->accept(this);
  }

  void generate_range_for_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::range_for;

    task_attribs_.range_for_attribs = TaskAttributes::RangeForAttributes();
    auto &range_for_attribs = task_attribs_.range_for_attribs.value();
    range_for_attribs.const_begin = stmt->const_begin;
    range_for_attribs.const_end = stmt->const_end;
    range_for_attribs.begin =
        (stmt->const_begin ? stmt->begin_value : stmt->begin_offset);
    range_for_attribs.end =
        (stmt->const_end ? stmt->end_value : stmt->end_offset);

    ir_->start_function(kernel_function_);
    const std::string total_elems_name("total_elems");
    spirv::Value total_elems;
    spirv::Value begin_expr_value;
    if (range_for_attribs.const_range()) {
      const int num_elems = range_for_attribs.end - range_for_attribs.begin;
      begin_expr_value = ir_->int_immediate_number(
          ir_->i32_type(), stmt->begin_value, false);  // Named Constant
      total_elems = ir_->int_immediate_number(ir_->i32_type(), num_elems,
                                              false);  // Named Constant
      task_attribs_.advisory_total_num_threads = num_elems;
    } else {
      spirv::Value end_expr_value;
      if (stmt->end_stmt) {
        // Range from args
        TI_ASSERT(stmt->const_begin);
        begin_expr_value = ir_->int_immediate_number(ir_->i32_type(),
                                                     stmt->begin_value, false);
        gen_array_range(stmt->end_stmt);
        end_expr_value = ir_->query_value(stmt->end_stmt->raw_name());
      } else {
        // Range from gtmp / constant
        if (!stmt->const_begin) {
          spirv::Value begin_idx = ir_->make_value(
              spv::OpShiftRightArithmetic, ir_->i32_type(),
              ir_->int_immediate_number(ir_->i32_type(), stmt->begin_offset),
              ir_->int_immediate_number(ir_->i32_type(), 2));
          begin_expr_value = ir_->load_variable(
              ir_->struct_array_access(
                  ir_->i32_type(),
                  get_buffer_value(BufferType::GlobalTmps, PrimitiveType::i32),
                  begin_idx),
              ir_->i32_type());
        } else {
          begin_expr_value = ir_->int_immediate_number(
              ir_->i32_type(), stmt->begin_value, false);  // Named Constant
        }
        if (!stmt->const_end) {
          spirv::Value end_idx = ir_->make_value(
              spv::OpShiftRightArithmetic, ir_->i32_type(),
              ir_->int_immediate_number(ir_->i32_type(), stmt->end_offset),
              ir_->int_immediate_number(ir_->i32_type(), 2));
          end_expr_value = ir_->load_variable(
              ir_->struct_array_access(
                  ir_->i32_type(),
                  get_buffer_value(BufferType::GlobalTmps, PrimitiveType::i32),
                  end_idx),
              ir_->i32_type());
        } else {
          end_expr_value =
              ir_->int_immediate_number(ir_->i32_type(), stmt->end_value, true);
        }
      }
      total_elems = ir_->sub(end_expr_value, begin_expr_value);
      task_attribs_.advisory_total_num_threads = kMaxNumThreadsGridStrideLoop;
    }
    task_attribs_.advisory_num_threads_per_group = stmt->block_dim;
    ir_->debug_name(spv::OpName, begin_expr_value, "begin_expr_value");
    ir_->debug_name(spv::OpName, total_elems, total_elems_name);

    spirv::Value begin_ =
        ir_->add(ir_->cast(ir_->i32_type(), ir_->get_global_invocation_id(0)),
                 begin_expr_value);
    ir_->debug_name(spv::OpName, begin_, "begin_");
    spirv::Value end_ = ir_->add(total_elems, begin_expr_value);
    ir_->debug_name(spv::OpName, end_, "end_");
    const std::string total_invocs_name = "total_invocs";
    // For now, |total_invocs_name| is equal to |total_elems|. Once we support
    // dynamic range, they will be different.
    // https://www.khronos.org/opengl/wiki/Compute_Shader#Inputs

    // HLSL & WGSL cross compilers do not support this builtin
    spirv::Value total_invocs = ir_->cast(
        ir_->i32_type(),
        ir_->mul(ir_->get_num_work_groups(0),
                 ir_->uint_immediate_number(
                     ir_->u32_type(),
                     task_attribs_.advisory_num_threads_per_group, true)));
    /*
    const int group_x = (task_attribs_.advisory_total_num_threads +
                         task_attribs_.advisory_num_threads_per_group - 1) /
                        task_attribs_.advisory_num_threads_per_group;
    spirv::Value total_invocs = ir_->uint_immediate_number(
        ir_->i32_type(), group_x * task_attribs_.advisory_num_threads_per_group,
        false);
        */

    ir_->debug_name(spv::OpName, total_invocs, total_invocs_name);

    // Must get init label after making value(to make sure they are correct)
    spirv::Label init_label = ir_->current_label();
    spirv::Label head_label = ir_->new_label();
    spirv::Label body_label = ir_->new_label();
    spirv::Label continue_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();
    ir_->make_inst(spv::OpBranch, head_label);

    // loop head
    ir_->start_label(head_label);
    spirv::PhiValue loop_var = ir_->make_phi(begin_.stype, 2);
    ir_->register_value("ii", loop_var);
    loop_var.set_incoming(0, begin_, init_label);
    spirv::Value loop_cond = ir_->lt(loop_var, end_);
    ir_->make_inst(spv::OpLoopMerge, merge_label, continue_label,
                   spv::LoopControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, loop_cond, body_label,
                   merge_label);

    // loop body
    ir_->start_label(body_label);
    push_loop_control_labels(continue_label, merge_label);

    // loop kernel
    stmt->body->accept(this);
    pop_loop_control_labels();
    ir_->make_inst(spv::OpBranch, continue_label);

    // loop continue
    ir_->start_label(continue_label);
    spirv::Value next_value = ir_->add(loop_var, total_invocs);
    loop_var.set_incoming(1, next_value, ir_->current_label());
    ir_->make_inst(spv::OpBranch, head_label);

    // loop merge
    ir_->start_label(merge_label);

    ir_->make_inst(spv::OpReturn);
    ir_->make_inst(spv::OpFunctionEnd);

    task_attribs_.buffer_binds = get_buffer_binds();
    task_attribs_.texture_binds = get_texture_binds();
  }

  void generate_struct_for_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::struct_for;
    task_attribs_.advisory_total_num_threads = 65536;
    task_attribs_.advisory_num_threads_per_group = 128;

    // The computation for a single work is wrapped inside a function, so that
    // we can do grid-strided loop.
    ir_->start_function(kernel_function_);

    auto listgen_buffer =
        get_buffer_value(BufferType::ListGen, PrimitiveType::u32);
    auto listgen_count_ptr = ir_->struct_array_access(
        ir_->u32_type(), listgen_buffer, ir_->const_i32_zero_);
    auto listgen_count = ir_->load_variable(listgen_count_ptr, ir_->u32_type());

    auto invoc_index = ir_->get_global_invocation_id(0);

    spirv::Label loop_head = ir_->new_label();
    spirv::Label loop_body = ir_->new_label();
    spirv::Label loop_merge = ir_->new_label();

    auto loop_index_var = ir_->alloca_variable(ir_->u32_type());
    ir_->store_variable(loop_index_var, invoc_index);

    ir_->make_inst(spv::OpBranch, loop_head);
    ir_->start_label(loop_head);
    // for (; index < list_size; index += gl_NumWorkGroups.x *
    // gl_WorkGroupSize.x)
    auto loop_index = ir_->load_variable(loop_index_var, ir_->u32_type());
    auto loop_cond = ir_->make_value(spv::OpULessThan, ir_->bool_type(),
                                     loop_index, listgen_count);
    ir_->make_inst(spv::OpLoopMerge, loop_merge, loop_body,
                   spv::LoopControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, loop_cond, loop_body, loop_merge);
    {
      ir_->start_label(loop_body);
      auto listgen_index_ptr = ir_->struct_array_access(
          ir_->u32_type(), listgen_buffer,
          ir_->add(ir_->uint_immediate_number(ir_->u32_type(), 1), loop_index));
      auto listgen_index =
          ir_->load_variable(listgen_index_ptr, ir_->u32_type());

      // kernel
      ir_->register_value("ii", listgen_index);
      stmt->body->accept(this);

      // continue
      spirv::Value total_invocs = ir_->cast(
          ir_->u32_type(),
          ir_->mul(ir_->get_num_work_groups(0),
                   ir_->uint_immediate_number(
                       ir_->u32_type(),
                       task_attribs_.advisory_num_threads_per_group, true)));
      auto next_index = ir_->add(loop_index, total_invocs);
      ir_->store_variable(loop_index_var, next_index);
      ir_->make_inst(spv::OpBranch, loop_head);
    }
    ir_->start_label(loop_merge);

    ir_->make_inst(spv::OpReturn);       // return;
    ir_->make_inst(spv::OpFunctionEnd);  // } Close kernel

    task_attribs_.buffer_binds = get_buffer_binds();
    task_attribs_.texture_binds = get_texture_binds();
  }

  spirv::Value at_buffer(const Stmt *ptr, DataType dt) {
    spirv::Value ptr_val = ir_->query_value(ptr->raw_name());

    if (ptr_val.stype.dt == PrimitiveType::u64) {
      spirv::Value paddr_ptr = ir_->make_value(
          spv::OpConvertUToPtr,
          ir_->get_pointer_type(ir_->get_primitive_type(dt),
                                spv::StorageClassPhysicalStorageBuffer),
          ptr_val);
      paddr_ptr.flag = ValueKind::kPhysicalPtr;
      return paddr_ptr;
    }

    spirv::Value buffer = get_buffer_value(ptr_to_buffers_.at(ptr), dt);
    size_t width = ir_->get_primitive_type_size(dt);
    spirv::Value idx_val = ir_->make_value(
        spv::OpShiftRightLogical, ptr_val.stype, ptr_val,
        ir_->uint_immediate_number(ptr_val.stype, size_t(std::log2(width))));
    spirv::Value ret =
        ir_->struct_array_access(ir_->get_primitive_type(dt), buffer, idx_val);
    return ret;
  }

  spirv::Value load_buffer(const Stmt *ptr, DataType dt) {
    spirv::Value ptr_val = ir_->query_value(ptr->raw_name());

    DataType ti_buffer_type = ir_->get_taichi_uint_type(dt);

    if (ptr_val.stype.dt == PrimitiveType::u64) {
      ti_buffer_type = dt;
    }

    auto buf_ptr = at_buffer(ptr, ti_buffer_type);
    auto val_bits =
        ir_->load_variable(buf_ptr, ir_->get_primitive_type(ti_buffer_type));
    auto ret = ti_buffer_type == dt
                   ? val_bits
                   : ir_->make_value(spv::OpBitcast,
                                     ir_->get_primitive_type(dt), val_bits);
    return ret;
  }

  void store_buffer(const Stmt *ptr, spirv::Value val) {
    spirv::Value ptr_val = ir_->query_value(ptr->raw_name());

    DataType ti_buffer_type = ir_->get_taichi_uint_type(val.stype.dt);

    if (ptr_val.stype.dt == PrimitiveType::u64) {
      ti_buffer_type = val.stype.dt;
    }

    auto buf_ptr = at_buffer(ptr, ti_buffer_type);
    auto val_bits =
        val.stype.dt == ti_buffer_type
            ? val
            : ir_->make_value(spv::OpBitcast,
                              ir_->get_primitive_type(ti_buffer_type), val);
    ir_->store_variable(buf_ptr, val_bits);
  }

  spirv::Value get_buffer_value(BufferInfo buffer, DataType dt) {
    auto type = ir_->get_primitive_type(dt);
    auto key = std::make_pair(buffer, type.id);

    const auto it = buffer_value_map_.find(key);
    if (it != buffer_value_map_.end()) {
      return it->second;
    }

    if (buffer.type == BufferType::Args) {
      compile_args_struct();

      buffer_binding_map_[key] = 0;
      buffer_value_map_[key] = args_buffer_value_;
      return args_buffer_value_;
    }

    if (buffer.type == BufferType::Rets) {
      compile_ret_struct();

      buffer_binding_map_[key] = 1;
      buffer_value_map_[key] = ret_buffer_value_;
      return ret_buffer_value_;
    }

    // Binding head starts at 2, so we don't break args and rets
    int binding = binding_head_++;
    buffer_binding_map_[key] = binding;

    spirv::Value buffer_value =
        ir_->buffer_argument(type, 0, binding, buffer_instance_name(buffer));
    buffer_value_map_[key] = buffer_value;
    TI_TRACE("buffer name = {}, value = {}", buffer_instance_name(buffer),
             buffer_value.id);

    return buffer_value;
  }

  spirv::Value make_pointer(size_t offset) {
    if (use_64bit_pointers) {
      // This is hacky, should check out how to encode uint64 values in spirv
      return ir_->cast(ir_->u64_type(), ir_->uint_immediate_number(
                                            ir_->u32_type(), uint32_t(offset)));
    } else {
      return ir_->uint_immediate_number(ir_->u32_type(), uint32_t(offset));
    }
  }

  void compile_args_struct() {
    if (!ctx_attribs_->has_args())
      return;

    // Generate struct IR
    tinyir::Block blk;
    std::vector<const tinyir::Type *> element_types;
    bool has_buffer_ptr =
        caps_->get(DeviceCapability::spirv_has_physical_storage_buffer);
    for (auto &element : ctx_attribs_->args_type()->elements()) {
      element_types.push_back(
          translate_ti_type(blk, element.type, has_buffer_ptr));
    }
    const tinyir::Type *i32_type =
        blk.emplace_back<IntType>(/*num_bits=*/32, /*is_signed=*/true);
    for (int i = 0; i < ctx_attribs_->extra_args_bytes() / 4; i++) {
      element_types.push_back(i32_type);
    }
    const tinyir::Type *struct_type =
        blk.emplace_back<StructType>(element_types);

    // Reduce struct IR
    std::unordered_map<const tinyir::Type *, const tinyir::Type *> old2new;
    auto reduced_blk = ir_reduce_types(&blk, old2new);
    struct_type = old2new[struct_type];

    for (auto &element : element_types) {
      element = old2new[element];
    }

    // Layout & translate to SPIR-V
    STD140LayoutContext layout_ctx;
    auto ir2spirv_map =
        ir_translate_to_spirv(reduced_blk.get(), layout_ctx, ir_.get());
    args_struct_type_.id = ir2spirv_map[struct_type];

    // Must use the same type in ArgLoadStmt as in the args struct,
    // otherwise the validation will fail.
    args_struct_types_.resize(element_types.size());
    for (int i = 0; i < element_types.size(); i++) {
      args_struct_types_[i].id = ir2spirv_map.at(element_types[i]);
      if (i < ctx_attribs_->args_type()->elements().size()) {
        args_struct_types_[i].dt =
            ctx_attribs_->args_type()->get_element_type({i});
      } else {
        args_struct_types_[i].dt = PrimitiveType::i32;
      }
    }

    args_buffer_value_ =
        ir_->uniform_struct_argument(args_struct_type_, 0, 0, "args");
  }

  void compile_ret_struct() {
    if (!ctx_attribs_->has_rets())
      return;

    std::vector<std::tuple<spirv::SType, std::string, size_t>>
        struct_components_;
    // Now we only have one ret
    TI_ASSERT(ctx_attribs_->rets().size() == 1);
    for (auto &ret : ctx_attribs_->rets()) {
      // Use array size = 0 to generate a RuntimeArray
      if (auto tensor_type =
              PrimitiveType::get(ret.dtype)->cast<TensorType>()) {
        struct_components_.emplace_back(
            ir_->get_array_type(
                ir_->get_primitive_type(tensor_type->get_element_type()), 0),
            "ret" + std::to_string(ret.index), ret.offset_in_mem);
      } else {
        struct_components_.emplace_back(
            ir_->get_array_type(
                ir_->get_primitive_type(PrimitiveType::get(ret.dtype)), 0),
            "ret" + std::to_string(ret.index), ret.offset_in_mem);
      }
    }
    ret_struct_type_ = ir_->create_struct_type(struct_components_);

    ret_buffer_value_ =
        ir_->buffer_struct_argument(ret_struct_type_, 0, 1, "rets");
  }

  std::vector<BufferBind> get_buffer_binds() {
    std::vector<BufferBind> result;
    for (auto &[key, val] : buffer_binding_map_) {
      result.push_back(BufferBind{key.first, int(val)});
    }
    return result;
  }

  std::vector<TextureBind> get_texture_binds() {
    return texture_binds_;
  }

  void push_loop_control_labels(spirv::Label continue_label,
                                spirv::Label merge_label) {
    continue_label_stack_.push_back(continue_label);
    merge_label_stack_.push_back(merge_label);
  }

  void pop_loop_control_labels() {
    continue_label_stack_.pop_back();
    merge_label_stack_.pop_back();
  }

  const spirv::Label current_continue_label() const {
    return continue_label_stack_.back();
  }

  const spirv::Label current_merge_label() const {
    return merge_label_stack_.back();
  }

  const spirv::Label return_label() const {
    return continue_label_stack_.front();
  }

  Arch arch_;
  DeviceCapabilityConfig *caps_;

  struct BufferInfoTypeTupleHasher {
    std::size_t operator()(const std::pair<BufferInfo, int> &buf) const {
      return BufferInfoHasher()(buf.first) ^ (buf.second << 5);
    }
  };

  spirv::SType args_struct_type_;
  spirv::Value args_buffer_value_;

  std::vector<spirv::SType> args_struct_types_;

  spirv::SType ret_struct_type_;
  spirv::Value ret_buffer_value_;

  std::shared_ptr<spirv::IRBuilder> ir_;  // spirv binary code builder
  std::unordered_map<std::pair<BufferInfo, int>,
                     spirv::Value,
                     BufferInfoTypeTupleHasher>
      buffer_value_map_;
  std::unordered_map<std::pair<BufferInfo, int>,
                     uint32_t,
                     BufferInfoTypeTupleHasher>
      buffer_binding_map_;
  std::vector<TextureBind> texture_binds_;
  std::vector<spirv::Value> shared_array_binds_;
  spirv::Value kernel_function_;
  spirv::Label kernel_return_label_;
  bool gen_label_{false};

  int binding_head_{2};  // Args:0, Ret:1

  /*
  std::unordered_map<int, spirv::CompiledSpirvSNode>
      spirv_snodes_;  // maps root id to spirv snode
      */

  OffloadedStmt *const task_ir_;  // not owned
  std::vector<CompiledSNodeStructs> compiled_structs_;
  std::unordered_map<int, int> snode_to_root_;
  const KernelContextAttributes *const ctx_attribs_;  // not owned
  const std::string task_name_;
  std::vector<spirv::Label> continue_label_stack_;
  std::vector<spirv::Label> merge_label_stack_;

  std::unordered_set<const Stmt *> offload_loop_motion_;

  TaskAttributes task_attribs_;
  std::unordered_map<int, GetRootStmt *>
      root_stmts_;  // maps root id to get root stmt
  std::unordered_map<const Stmt *, BufferInfo> ptr_to_buffers_;
  std::unordered_map<int, Value> argid_to_tex_value_;
};
}  // namespace

static void spriv_message_consumer(spv_message_level_t level,
                                   const char *source,
                                   const spv_position_t &position,
                                   const char *message) {
  // TODO: Maybe we can add a macro, e.g. TI_LOG_AT_LEVEL(lv, ...)
  if (level <= SPV_MSG_FATAL) {
    TI_ERROR("{}\n[{}:{}:{}] {}", source, position.index, position.line,
             position.column, message);
  } else if (level <= SPV_MSG_WARNING) {
    TI_WARN("{}\n[{}:{}:{}] {}", source, position.index, position.line,
            position.column, message);
  } else if (level <= SPV_MSG_INFO) {
    TI_INFO("{}\n[{}:{}:{}] {}", source, position.index, position.line,
            position.column, message);
  } else if (level <= SPV_MSG_INFO) {
    TI_TRACE("{}\n[{}:{}:{}] {}", source, position.index, position.line,
             position.column, message);
  }
}

KernelCodegen::KernelCodegen(const Params &params)
    : params_(params), ctx_attribs_(*params.kernel, &params.caps) {
  TI_ASSERT(params.kernel);
  TI_ASSERT(params.ir_root);

  uint32_t spirv_version = params.caps.get(DeviceCapability::spirv_version);

  spv_target_env target_env;
  if (spirv_version >= 0x10600) {
    target_env = SPV_ENV_VULKAN_1_3;
  } else if (spirv_version >= 0x10500) {
    target_env = SPV_ENV_VULKAN_1_2;
  } else if (spirv_version >= 0x10400) {
    target_env = SPV_ENV_VULKAN_1_1_SPIRV_1_4;
  } else if (spirv_version >= 0x10300) {
    target_env = SPV_ENV_VULKAN_1_1;
  } else {
    target_env = SPV_ENV_VULKAN_1_0;
  }

  spirv_opt_ = std::make_unique<spvtools::Optimizer>(target_env);
  spirv_opt_->SetMessageConsumer(spriv_message_consumer);
  if (params.enable_spv_opt) {
    // From: SPIRV-Tools/source/opt/optimizer.cpp
    spirv_opt_->RegisterPass(spvtools::CreateWrapOpKillPass())
        .RegisterPass(spvtools::CreateDeadBranchElimPass())
        .RegisterPass(spvtools::CreateMergeReturnPass())
        .RegisterPass(spvtools::CreateInlineExhaustivePass())
        .RegisterPass(spvtools::CreateEliminateDeadFunctionsPass())
        .RegisterPass(spvtools::CreateAggressiveDCEPass())
        .RegisterPass(spvtools::CreatePrivateToLocalPass())
        .RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass())
        .RegisterPass(spvtools::CreateLocalSingleStoreElimPass())
        .RegisterPass(spvtools::CreateScalarReplacementPass())
        .RegisterPass(spvtools::CreateLocalAccessChainConvertPass())
        .RegisterPass(spvtools::CreateLocalMultiStoreElimPass())
        .RegisterPass(spvtools::CreateCCPPass())
        .RegisterPass(spvtools::CreateLoopUnrollPass(true))
        .RegisterPass(spvtools::CreateRedundancyEliminationPass())
        .RegisterPass(spvtools::CreateCombineAccessChainsPass())
        .RegisterPass(spvtools::CreateSimplificationPass())
        .RegisterPass(spvtools::CreateSSARewritePass())
        .RegisterPass(spvtools::CreateVectorDCEPass())
        .RegisterPass(spvtools::CreateDeadInsertElimPass())
        .RegisterPass(spvtools::CreateIfConversionPass())
        .RegisterPass(spvtools::CreateCopyPropagateArraysPass())
        .RegisterPass(spvtools::CreateReduceLoadSizePass())
        .RegisterPass(spvtools::CreateBlockMergePass());
  }
  spirv_opt_options_.set_run_validator(false);

  spirv_tools_ = std::make_unique<spvtools::SpirvTools>(target_env);
}

void KernelCodegen::run(TaichiKernelAttributes &kernel_attribs,
                        std::vector<std::vector<uint32_t>> &generated_spirv) {
  auto *root = params_.ir_root->as<Block>();
  auto &tasks = root->statements;
  for (int i = 0; i < tasks.size(); ++i) {
    TaskCodegen::Params tp;
    tp.task_ir = tasks[i]->as<OffloadedStmt>();
    tp.task_id_in_kernel = i;
    tp.compiled_structs = params_.compiled_structs;
    tp.ctx_attribs = &ctx_attribs_;
    tp.ti_kernel_name = fmt::format("{}_{}", params_.ti_kernel_name, i);
    tp.arch = params_.arch;
    tp.caps = &params_.caps;

    TaskCodegen cgen(tp);
    auto task_res = cgen.run();

    for (auto &[id, access] : task_res.arr_access) {
      ctx_attribs_.arr_access[id] = ctx_attribs_.arr_access[id] | access;
    }

    std::vector<uint32_t> optimized_spv(task_res.spirv_code);

    bool success = true;
    {
      bool result = false;
      TI_ERROR_IF(
          (result = !spirv_opt_->Run(optimized_spv.data(), optimized_spv.size(),
                                     &optimized_spv, spirv_opt_options_)),
          "SPIRV optimization failed");
      if (result) {
        success = false;
      }
    }

    TI_TRACE("SPIRV-Tools-opt: binary size, before={}, after={}",
             task_res.spirv_code.size(), optimized_spv.size());

    // Enable to dump SPIR-V assembly of kernels
    if constexpr (false) {
      std::vector<uint32_t> &spirv =
          success ? optimized_spv : task_res.spirv_code;

      std::string spirv_asm;
      spirv_tools_->Disassemble(optimized_spv, &spirv_asm);
      auto kernel_name = tp.ti_kernel_name;
      TI_WARN("SPIR-V Assembly dump for {} :\n{}\n\n", kernel_name, spirv_asm);

      std::ofstream fout(kernel_name + ".spv",
                         std::ios::binary | std::ios::out);
      fout.write(reinterpret_cast<const char *>(spirv.data()),
                 spirv.size() * sizeof(uint32_t));
      fout.close();
    }

    kernel_attribs.tasks_attribs.push_back(std::move(task_res.task_attribs));
    generated_spirv.push_back(std::move(optimized_spv));
  }
  kernel_attribs.ctx_attribs = std::move(ctx_attribs_);
  kernel_attribs.name = params_.ti_kernel_name;
}

}  // namespace spirv
}  // namespace taichi::lang
