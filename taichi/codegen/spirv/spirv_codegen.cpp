#include "taichi/codegen/spirv/spirv_codegen.h"

#include <string>
#include <vector>

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

namespace taichi {
namespace lang {
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
    Device *device;
    std::vector<CompiledSNodeStructs> compiled_structs;
    const KernelContextAttributes *ctx_attribs;
    std::string ti_kernel_name;
    int task_id_in_kernel;
  };

  const bool use_64bit_pointers = false;

  explicit TaskCodegen(const Params &params)
      : device_(params.device),
        task_ir_(params.task_ir),
        compiled_structs_(params.compiled_structs),
        ctx_attribs_(params.ctx_attribs),
        task_name_(fmt::format("{}_t{:02d}",
                               params.ti_kernel_name,
                               params.task_id_in_kernel)) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;

    fill_snode_to_root();
    ir_ = std::make_shared<spirv::IRBuilder>(params.device);
  }

  void fill_snode_to_root() {
    for (int root = 0; root < compiled_structs_.size(); ++root) {
      for (auto [node_id, node] : compiled_structs_[root].snode_descriptors) {
        snode_to_root_[node_id] = root;
      }
    }
  }

  struct Result {
    std::vector<uint32_t> spirv_code;
    TaskAttributes task_attribs;
    std::unordered_map<int, irpass::ExternalPtrAccess> arr_access;
  };

  Result run() {
    ir_->init_header();
    kernel_function_ = ir_->new_function();  // void main();
    ir_->debug(spv::OpName, kernel_function_, "main");

    compile_args_struct();
    compile_ret_struct();

    if (task_ir_->task_type == OffloadedTaskType::serial) {
      generate_serial_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::range_for) {
      // struct_for is automatically lowered to ranged_for for dense snodes
      generate_range_for_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::listgen) {
      generate_listgen_kernel(task_ir_);
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

  void visit(ConstStmt *const_stmt) override {
    TI_ASSERT(const_stmt->width() == 1);

    auto get_const = [&](const TypedConstant &const_val) {
      auto dt = const_val.dt.ptr_removed();
      spirv::SType stype = ir_->get_primitive_type(dt);

      if (dt->is_primitive(PrimitiveTypeID::f32)) {
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

    spirv::Value val = get_const(const_stmt->val[0]);
    ir_->register_value(const_stmt->raw_name(), val);
  }

  void visit(AllocaStmt *alloca) override {
    spirv::SType src_type = ir_->get_primitive_type(alloca->element_type());
    spirv::Value ptr_val = ir_->alloca_variable(src_type);
    ir_->store_variable(ptr_val, ir_->get_zero(src_type));
    ir_->register_value(alloca->raw_name(), ptr_val);
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
      spirv::Value ptr_val = ir_->query_value(ptr->raw_name());
      spirv::Value val = ir_->load_variable(
          ptr_val, ir_->get_primitive_type(stmt->element_type()));
      ir_->register_value(stmt->raw_name(), val);
    } else {
      TI_NOT_IMPLEMENTED
    }
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
        make_pointer(desc.cell_stride * desc.cells_per_container_pot()));
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

  void visit(BitExtractStmt *stmt) override {
    spirv::Value input_val = ir_->query_value(stmt->input->raw_name());
    auto stype = input_val.stype;
    spirv::Value tmp0 = ir_->int_immediate_number(stype, stmt->bit_begin);
    spirv::Value tmp1 =
        ir_->int_immediate_number(stype, stmt->bit_end - stmt->bit_begin);
    spirv::Value tmp2 =
        ir_->make_value(spv::OpShiftRightArithmetic, stype, input_val, tmp0);
    spirv::Value tmp3 =
        ir_->make_value(spv::OpShiftLeftLogical, stype,
                        ir_->int_immediate_number(stype, 1), tmp1);
    spirv::Value tmp4 = ir_->sub(tmp3, ir_->int_immediate_number(stype, 1));
    spirv::Value val = ir_->make_value(spv::OpBitwiseAnd, stype, tmp2, tmp4);
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
      } else if (type == OffloadedTaskType::struct_for) {
        SNode *snode = stmt->loop->as<OffloadedStmt>()->snode;
        spirv::Value val = ir_->query_value("ii");
        // FIXME: packed layout (non POT)
        int root_id = snode_to_root_[snode->id];
        const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
        const int *axis_start_bit = snode_descs.at(snode->id).axis_start_bit;
        const int *axis_bits_sum = snode_descs.at(snode->id).axis_bits_sum;
        val =
            ir_->make_value(spv::OpShiftRightLogical, ir_->u32_type(), val,
                            ir_->uint_immediate_number(
                                ir_->u32_type(), axis_start_bit[stmt->index]));
        val = ir_->make_value(
            spv::OpBitwiseAnd, ir_->u32_type(), val,
            ir_->uint_immediate_number(ir_->u32_type(),
                                       (1 << axis_bits_sum[stmt->index]) - 1));
        val = ir_->cast(ir_->i32_type(), val);
        ir_->register_value(stmt_name, val);
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
    TI_ASSERT(stmt->width() == 1);

    spirv::Value val = ir_->query_value(stmt->val->raw_name());

    store_buffer(stmt->dest, val);
  }

  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt = stmt->element_type();

    auto val = load_buffer(stmt->src, dt);

    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(ArgLoadStmt *stmt) override {
    const auto arg_id = stmt->arg_id;
    const auto &arg_attribs = ctx_attribs_->args()[arg_id];
    if (stmt->is_ptr) {
      // Do not shift! We are indexing the buffers at byte granularity.
      // spirv::Value val =
      //    ir_->int_immediate_number(ir_->i32_type(), offset_in_mem);
      // ir_->register_value(stmt->raw_name(), val);
    } else {
      const auto dt = PrimitiveType::get(arg_attribs.dtype);
      const auto val_type = ir_->get_primitive_type(dt);
      spirv::Value buffer_val = ir_->make_value(
          spv::OpAccessChain,
          ir_->get_pointer_type(val_type, spv::StorageClassUniform),
          get_buffer_value(BufferType::Args, PrimitiveType::i32),
          ir_->int_immediate_number(ir_->i32_type(), arg_id));
      buffer_val.flag = ValueKind::kVariablePtr;
      spirv::Value val = ir_->load_variable(buffer_val, val_type);
      ir_->register_value(stmt->raw_name(), val);
    }
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
    TI_ASSERT(stmt->width() == 1);
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
    spirv::Value var_ptr = ir_->make_value(
        spv::OpAccessChain,
        ir_->get_pointer_type(ir_->i32_type(), spv::StorageClassUniform),
        get_buffer_value(BufferType::Args, PrimitiveType::i32),
        ir_->int_immediate_number(ir_->i32_type(),
                                  extra_args_member_index + extra_arg_index));
    spirv::Value var = ir_->load_variable(var_ptr, ir_->i32_type());

    ir_->register_value(name, var);
  }

  void visit(ExternalPtrStmt *stmt) override {
    // Used mostly for transferring data between host (e.g. numpy array) and
    // device.
    TI_ASSERT(stmt->width() == 1);
    spirv::Value linear_offset = ir_->int_immediate_number(ir_->i32_type(), 0);
    const auto *argload = stmt->base_ptrs[0]->as<ArgLoadStmt>();
    const int arg_id = argload->arg_id;
    {
      const int num_indices = stmt->indices.size();
      std::vector<std::string> size_var_names;
      const auto &element_shape = stmt->element_shape;
      const auto layout = stmt->element_dim <= 0 ? ExternalArrayLayout::kAOS
                                                 : ExternalArrayLayout::kSOA;
      const auto extra_args_member_index = ctx_attribs_->args().size();
      const size_t element_shape_index_offset =
          (layout == ExternalArrayLayout::kAOS)
              ? num_indices - element_shape.size()
              : 0;
      for (int i = 0; i < num_indices - element_shape.size(); i++) {
        std::string var_name = fmt::format("{}_size{}_", stmt->raw_name(), i);
        const auto extra_arg_index = (arg_id * taichi_max_num_indices) + i;
        spirv::Value var_ptr = ir_->make_value(
            spv::OpAccessChain,
            ir_->get_pointer_type(ir_->i32_type(), spv::StorageClassUniform),
            get_buffer_value(BufferType::Args, PrimitiveType::i32),
            ir_->int_immediate_number(
                ir_->i32_type(), extra_args_member_index + extra_arg_index));
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
                                        argload->ret_type.ptr_removed()))));
      ir_->decorate(spv::OpDecorate, linear_offset,
                    spv::DecorationNoSignedWrap);
    }
    if (device_->get_cap(DeviceCapability::spirv_has_physical_storage_buffer)) {
      spirv::Value addr_ptr = ir_->make_value(
          spv::OpAccessChain,
          ir_->get_pointer_type(ir_->u64_type(), spv::StorageClassUniform),
          get_buffer_value(BufferType::Args, PrimitiveType::i32),
          ir_->int_immediate_number(ir_->i32_type(), arg_id));
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
    spirv::SType dst_type = ir_->get_primitive_type(dst_dt);
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

    if (false) {
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

    BINARY_OP_TO_SPIRV_BITWISE(bit_and, OpBitwiseAnd)
    BINARY_OP_TO_SPIRV_BITWISE(bit_or, OpBitwiseOr)
    BINARY_OP_TO_SPIRV_BITWISE(bit_xor, OpBitwiseXor)
    BINARY_OP_TO_SPIRV_BITWISE(bit_shl, OpShiftLeftLogical)
    BINARY_OP_TO_SPIRV_BITWISE(bit_shr, OpShiftRightLogical)
    // NOTE: `OpShiftRightArithmetic` will treat the first bit as sign bit even
    // it's the unsigned type
    else if (op_type == BinaryOpType::bit_sar) {
      bin_value = ir_->make_value(is_unsigned(dst_type.dt)
                                      ? spv::OpShiftRightLogical
                                      : spv::OpShiftRightArithmetic,
                                  dst_type, lhs_value, rhs_value);
    }
#undef BINARY_OP_TO_SPIRV_BITWISE

#define BINARY_OP_TO_SPIRV_LOGICAL(op, func)                          \
  else if (op_type == BinaryOpType::op) {                             \
    bin_value = ir_->func(lhs_value, rhs_value);                      \
    bin_value = ir_->cast(dst_type, bin_value);                       \
    bin_value = ir_->make_value(spv::OpSNegate, dst_type, bin_value); \
  }

    BINARY_OP_TO_SPIRV_LOGICAL(cmp_lt, lt)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_le, le)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_gt, gt)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_ge, ge)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_eq, eq)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_ne, ne)
#undef BINARY_OP_TO_SPIRV_LOGICAL

#define INT_OR_FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC(op, instruction,            \
                                                   instruction_id, max_bits)   \
  else if (op_type == BinaryOpType::op) {                                      \
    const uint32_t instruction = instruction_id;                               \
    if (is_real(bin->element_type()) || is_integral(bin->element_type())) {    \
      if (data_type_bits(bin->element_type()) > max_bits) {                    \
        TI_ERROR(                                                              \
            "[glsl450] the operand type of instruction {}({}) must <= {}bits", \
            #instruction, instruction_id, max_bits);                           \
      }                                                                        \
      if (is_integral(bin->element_type())) {                                  \
        bin_value = ir_->cast(                                                 \
            dst_type,                                                          \
            ir_->add(ir_->call_glsl450(ir_->f32_type(), instruction,           \
                                       ir_->cast(ir_->f32_type(), lhs_value),  \
                                       ir_->cast(ir_->f32_type(), rhs_value)), \
                     ir_->float_immediate_number(ir_->f32_type(), 0.5f)));     \
      } else {                                                                 \
        bin_value =                                                            \
            ir_->call_glsl450(dst_type, instruction, lhs_value, rhs_value);    \
      }                                                                        \
    } else {                                                                   \
      TI_NOT_IMPLEMENTED                                                       \
    }                                                                          \
  }

    INT_OR_FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC(pow, Pow, 26, 32)
#undef INT_OR_FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC

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
    else if (op_type == BinaryOpType::floordiv) {
      uint32_t Floor_id = 8;
      lhs_value =
          ir_->cast(ir_->f32_type(), lhs_value);  // TODO: Hard-coded f32
      rhs_value = ir_->cast(ir_->f32_type(), rhs_value);
      bin_value = ir_->div(lhs_value, rhs_value);
      bin_value = ir_->call_glsl450(ir_->f32_type(), Floor_id, bin_value);
      bin_value = ir_->cast(dst_type, bin_value);
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
        BufferFormat format = BufferFormat::unknown;

        if (stmt->num_channels == 1) {
          if (stmt->channel_format == PrimitiveType::u8 ||
              stmt->channel_format == PrimitiveType::i8) {
            format = BufferFormat::r8;
          } else if (stmt->channel_format == PrimitiveType::u16 ||
                     stmt->channel_format == PrimitiveType::i16) {
            format = BufferFormat::r16;
          } else if (stmt->channel_format == PrimitiveType::f16) {
            format = BufferFormat::r16f;
          } else if (stmt->channel_format == PrimitiveType::f32) {
            format = BufferFormat::r32f;
          }
        } else if (stmt->num_channels == 2) {
          if (stmt->channel_format == PrimitiveType::u8 ||
              stmt->channel_format == PrimitiveType::i8) {
            format = BufferFormat::rg8;
          } else if (stmt->channel_format == PrimitiveType::u16 ||
                     stmt->channel_format == PrimitiveType::i16) {
            format = BufferFormat::rg16;
          } else if (stmt->channel_format == PrimitiveType::f16) {
            format = BufferFormat::rg16f;
          } else if (stmt->channel_format == PrimitiveType::f32) {
            format = BufferFormat::rg32f;
          }
        } else if (stmt->num_channels == 4) {
          if (stmt->channel_format == PrimitiveType::u8 ||
              stmt->channel_format == PrimitiveType::i8) {
            format = BufferFormat::rgba8;
          } else if (stmt->channel_format == PrimitiveType::u16 ||
                     stmt->channel_format == PrimitiveType::i16) {
            format = BufferFormat::rgba16;
          } else if (stmt->channel_format == PrimitiveType::f16) {
            format = BufferFormat::rgba16f;
          } else if (stmt->channel_format == PrimitiveType::f32) {
            format = BufferFormat::rgba32f;
          }
        }

        int binding = binding_head_++;
        val = ir_->storage_image_argument(/*num_channels=*/4, stmt->dimensions,
                                          /*set=*/0, binding, format);
        TextureBind bind;
        bind.arg_id = arg_id;
        bind.binding = binding;
        bind.is_storage = true;
        texture_binds_.push_back(bind);
        argid_to_tex_value_[arg_id] = val;
      } else {
        int binding = binding_head_++;
        val = ir_->texture_argument(/*num_channels=*/4, stmt->dimensions,
                                    /*set=*/0, binding);
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

    if (stmt->func_name == "subgroupElect") {
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
    }
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(AtomicOpStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    const auto dt = stmt->dest->element_type().ptr_removed();

    spirv::Value data = ir_->query_value(stmt->val->raw_name());
    spirv::Value val;
    bool use_subgroup_reduction = false;

    if (stmt->is_reduction &&
        device_->get_cap(DeviceCapability::spirv_has_subgroup_arithmetic)) {
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
      if (device_->get_cap(DeviceCapability::spirv_has_atomic_float64_add) &&
          stmt->op_type == AtomicOpType::add) {
        addr_ptr = at_buffer(stmt->dest, dt);
      } else {
        addr_ptr = at_buffer(stmt->dest, ir_->get_taichi_uint_type(dt));
      }
    } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
      if (device_->get_cap(DeviceCapability::spirv_has_atomic_float_add) &&
          stmt->op_type == AtomicOpType::add) {
        addr_ptr = at_buffer(stmt->dest, dt);
      } else {
        addr_ptr = at_buffer(stmt->dest, ir_->get_taichi_uint_type(dt));
      }
    } else {
      addr_ptr = at_buffer(stmt->dest, dt);
    }

    auto ret_type = ir_->get_primitive_type(dt);

    if (is_real(dt)) {
      spv::Op atomic_fp_op;
      if (stmt->op_type == AtomicOpType::add) {
        atomic_fp_op = spv::OpAtomicFAddEXT;
      }

      bool use_native_atomics = false;

      if (dt->is_primitive(PrimitiveTypeID::f64)) {
        if (device_->get_cap(DeviceCapability::spirv_has_atomic_float64_add) &&
            stmt->op_type == AtomicOpType::add) {
          use_native_atomics = true;
        }
      } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
        if (device_->get_cap(DeviceCapability::spirv_has_atomic_float_add) &&
            stmt->op_type == AtomicOpType::add) {
          use_native_atomics = true;
        }
      } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
        if (device_->get_cap(DeviceCapability::spirv_has_atomic_float16_add) &&
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
        val = ir_->float_atomic(stmt->op_type, addr_ptr, data);
      }
    } else if (is_integral(dt)) {
      spv::Op op;
      if (stmt->op_type == AtomicOpType::add) {
        op = spv::OpAtomicIAdd;
      } else if (stmt->op_type == AtomicOpType::sub) {
        op = spv::OpAtomicISub;
      } else if (stmt->op_type == AtomicOpType::min) {
        op = is_signed(dt) ? spv::OpAtomicSMin : spv::OpAtomicUMin;
      } else if (stmt->op_type == AtomicOpType::max) {
        op = is_signed(dt) ? spv::OpAtomicSMax : spv::OpAtomicUMax;
      } else if (stmt->op_type == AtomicOpType::bit_or) {
        op = spv::OpAtomicOr;
      } else if (stmt->op_type == AtomicOpType::bit_and) {
        op = spv::OpAtomicAnd;
      } else if (stmt->op_type == AtomicOpType::bit_xor) {
        op = spv::OpAtomicXor;
      } else {
        TI_NOT_IMPLEMENTED
      }

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
    TI_ASSERT(for_stmt->width() == 1);
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
    if (device_->get_cap(DeviceCapability::spirv_version) > 0x10300) {
      for (const auto &bb : task_attribs_.buffer_binds) {
        for (auto &it : buffer_value_map_) {
          if (it.first.first == bb.buffer) {
            buffers.push_back(it.second);
          }
        }
      }
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
    ir_->debug(spv::OpName, begin_expr_value, "begin_expr_value");
    ir_->debug(spv::OpName, total_elems, total_elems_name);

    spirv::Value begin_ =
        ir_->add(ir_->cast(ir_->i32_type(), ir_->get_global_invocation_id(0)),
                 begin_expr_value);
    ir_->debug(spv::OpName, begin_, "begin_");
    spirv::Value end_ = ir_->add(total_elems, begin_expr_value);
    ir_->debug(spv::OpName, end_, "end_");
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

    ir_->debug(spv::OpName, total_invocs, total_invocs_name);

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

  void generate_listgen_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::listgen;
    task_attribs_.advisory_total_num_threads = 1;
    task_attribs_.advisory_num_threads_per_group = 32;

    auto snode = stmt->snode;

    TI_TRACE("Listgen for {}", snode->get_name());

    std::vector<SNode *> snode_path;
    std::vector<int> snode_path_num_cells;
    std::vector<std::array<int, taichi_max_num_indices>>
        snode_path_index_start_bit;
    int total_num_cells = 1;
    int root_id = 0;
    {
      // Construct the SNode path to the chosen node
      auto snode_head = snode;
      std::array<int, taichi_max_num_indices> start_indices{0};
      do {
        snode_path.push_back(snode_head);
        snode_path_num_cells.push_back(total_num_cells);
        snode_path_index_start_bit.push_back(start_indices);
        total_num_cells *= snode_head->num_cells_per_container;
        root_id = snode_head->id;
        for (int i = 0; i < taichi_max_num_indices; i++) {
          start_indices[i] += snode_head->extractors[i].num_bits;
        }
      } while ((snode_head = snode_head->parent));
    }

    const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
    const auto sn_desc = snode_descs.at(snode->id);

    for (int i = snode_path.size() - 1; i >= 0; i--) {
      const auto &desc = snode_descs.at(snode_path[i]->id);
      TI_TRACE("- {} ({})", snode_path[i]->get_name(),
               snode_path[i]->type_name());
      TI_TRACE("  is_place: {}, num_axis: {}, num_cells: {}",
               snode_path[i]->is_place(), snode_path[i]->num_active_indices,
               desc.cells_per_container_pot());
    }

    ir_->start_function(kernel_function_);

    if (snode->type == SNodeType::bitmasked) {
      task_attribs_.advisory_total_num_threads = total_num_cells;
      int num_cells = snode->num_cells_per_container;

      TI_INFO("ListGen {} * {}", total_num_cells / num_cells, num_cells);

      auto listgen_buffer =
          get_buffer_value(BufferInfo(BufferType::ListGen), PrimitiveType::i32);
      auto invoc_index = ir_->get_global_invocation_id(0);

      auto container_ptr = make_pointer(0);
      std::vector<spirv::Value> linear_indices(snode_path.size());
      for (int i = snode_path.size() - 1; i >= 0; i--) {
        // Offset the ptr to the cell on layer up
        SNode *this_snode = snode_path[i];
        const auto &this_snode_desc = snode_descs.at(this_snode->id);

        auto snode_linear_index =
            ir_->uint_immediate_number(ir_->u32_type(), 0);
        if (this_snode->num_active_indices) {
          for (int idx = 0; idx < taichi_max_num_indices; idx++) {
            if (this_snode->extractors[idx].active) {
              auto axis_local_index = ir_->make_value(
                  spv::OpShiftRightLogical, ir_->u32_type(), invoc_index,
                  ir_->uint_immediate_number(
                      ir_->u32_type(), sn_desc.axis_start_bit[idx] +
                                           snode_path_index_start_bit[i][idx]));
              axis_local_index = ir_->make_value(
                  spv::OpBitwiseAnd, ir_->u32_type(), axis_local_index,
                  ir_->uint_immediate_number(
                      ir_->u32_type(),
                      (1 << this_snode->extractors[idx].num_bits) - 1));
              snode_linear_index = ir_->make_value(
                  spv::OpBitwiseOr, ir_->u32_type(),
                  ir_->make_value(spv::OpShiftLeftLogical, ir_->u32_type(),
                                  snode_linear_index,
                                  ir_->uint_immediate_number(
                                      ir_->u32_type(),
                                      this_snode->extractors[idx].num_bits)),
                  axis_local_index);
            }
          }
        }

        if (i > 0) {
          const auto &next_snode_desc = snode_descs.at(snode_path[i - 1]->id);
          if (this_snode->num_active_indices) {
            container_ptr = ir_->add(
                container_ptr,
                ir_->mul(snode_linear_index,
                         ir_->uint_immediate_number(
                             ir_->u32_type(), this_snode_desc.cell_stride)));
          } else {
            container_ptr = ir_->add(
                container_ptr,
                make_pointer(next_snode_desc.mem_offset_in_parent_cell));
          }
        }

        linear_indices[i] = snode_linear_index;
      }

      // Check current bitmask mask within the cell
      auto index_is_active =
          bitmasked_activation(ActivationOp::query, container_ptr, root_id,
                               snode, linear_indices[0]);

      auto if_then_label = ir_->new_label();
      auto if_merge_label = ir_->new_label();

      ir_->make_inst(spv::OpSelectionMerge, if_merge_label,
                     spv::SelectionControlMaskNone);
      ir_->make_inst(spv::OpBranchConditional, index_is_active, if_then_label,
                     if_merge_label);
      // if (is_active)
      {
        ir_->start_label(if_then_label);

        auto listgen_count_ptr = ir_->struct_array_access(
            ir_->u32_type(), listgen_buffer, ir_->const_i32_zero_);
        auto index_count = ir_->make_value(
            spv::OpAtomicIAdd, ir_->u32_type(), listgen_count_ptr,
            /*scope=*/ir_->const_i32_one_,
            /*semantics=*/ir_->const_i32_zero_,
            ir_->uint_immediate_number(ir_->u32_type(), 1));
        auto listgen_index_ptr = ir_->struct_array_access(
            ir_->u32_type(), listgen_buffer,
            ir_->add(ir_->uint_immediate_number(ir_->u32_type(), 1),
                     index_count));
        ir_->store_variable(listgen_index_ptr, invoc_index);
        ir_->make_inst(spv::OpBranch, if_merge_label);
      }
      ir_->start_label(if_merge_label);
    } else if (snode->type == SNodeType::dense) {
      // Why??
    } else {
      TI_NOT_IMPLEMENTED;
    }

    ir_->make_inst(spv::OpReturn);       // return;
    ir_->make_inst(spv::OpFunctionEnd);  // } Close kernel

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
          ir_->i32_type(),
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
      buffer_binding_map_[key] = 0;
      buffer_value_map_[key] = args_buffer_value_;
      return args_buffer_value_;
    }

    if (buffer.type == BufferType::Rets) {
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
    for (auto &arg : ctx_attribs_->args()) {
      const tinyir::Type *t;
      if (arg.is_array &&
          device_->get_cap(
              DeviceCapability::spirv_has_physical_storage_buffer)) {
        t = blk.emplace_back<IntType>(/*num_bits=*/64, /*is_signed=*/false);
      } else {
        t = translate_ti_primitive(blk, PrimitiveType::get(arg.dtype));
      }
      element_types.push_back(t);
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

    // Layout & translate to SPIR-V
    STD140LayoutContext layout_ctx;
    auto ir2spirv_map =
        ir_translate_to_spirv(reduced_blk.get(), layout_ctx, ir_.get());
    args_struct_type_.id = ir2spirv_map[struct_type];

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

  Device *device_;

  struct BufferInfoTypeTupleHasher {
    std::size_t operator()(const std::pair<BufferInfo, int> &buf) const {
      return BufferInfoHasher()(buf.first) ^ (buf.second << 5);
    }
  };

  spirv::SType args_struct_type_;
  spirv::Value args_buffer_value_;

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
    : params_(params), ctx_attribs_(*params.kernel, params.device) {
  spv_target_env target_env = SPV_ENV_VULKAN_1_0;
  uint32_t spirv_version =
      params.device->get_cap(DeviceCapability::spirv_version);

  if (spirv_version >= 0x10600) {
    target_env = SPV_ENV_VULKAN_1_3;
  } else if (spirv_version >= 0x10500) {
    target_env = SPV_ENV_VULKAN_1_2;
  } else if (spirv_version >= 0x10400) {
    target_env = SPV_ENV_VULKAN_1_1_SPIRV_1_4;
  } else if (spirv_version >= 0x10300) {
    target_env = SPV_ENV_VULKAN_1_1;
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
  auto *root = params_.kernel->ir->as<Block>();
  auto &tasks = root->statements;
  for (int i = 0; i < tasks.size(); ++i) {
    TaskCodegen::Params tp;
    tp.task_ir = tasks[i]->as<OffloadedStmt>();
    tp.task_id_in_kernel = i;
    tp.compiled_structs = params_.compiled_structs;
    tp.ctx_attribs = &ctx_attribs_;
    tp.ti_kernel_name = fmt::format("{}_{}", params_.ti_kernel_name, i);
    tp.device = params_.device;

    TaskCodegen cgen(tp);
    auto task_res = cgen.run();

    for (auto &[id, access] : task_res.arr_access) {
      ctx_attribs_.arr_access[id] = ctx_attribs_.arr_access[id] | access;
    }

    std::vector<uint32_t> optimized_spv(task_res.spirv_code);

    size_t last_size;
    do {
      last_size = optimized_spv.size();
      bool result = false;
      TI_WARN_IF(
          (result = !spirv_opt_->Run(optimized_spv.data(), optimized_spv.size(),
                                     &optimized_spv, spirv_opt_options_)),
          "SPIRV optimization failed");
      if (result)
        break;
    } while (last_size != optimized_spv.size());

    TI_TRACE("SPIRV-Tools-opt: binary size, before={}, after={}",
             task_res.spirv_code.size(), optimized_spv.size());

    // Enable to dump SPIR-V assembly of kernels
#if 0
    std::string spirv_asm;
    spirv_tools_->Disassemble(optimized_spv, &spirv_asm);
    auto kernel_name = tp.ti_kernel_name;
    TI_WARN("SPIR-V Assembly dump for {} :\n{}\n\n", kernel_name, spirv_asm);

    std::ofstream fout(kernel_name + ".spv", std::ios::binary | std::ios::out);
    fout.write(reinterpret_cast<const char *>(optimized_spv.data()),
               optimized_spv.size() * sizeof(uint32_t));
    fout.close();
#endif

    kernel_attribs.tasks_attribs.push_back(std::move(task_res.task_attribs));
    generated_spirv.push_back(std::move(optimized_spv));
  }
  kernel_attribs.ctx_attribs = std::move(ctx_attribs_);
  kernel_attribs.name = params_.ti_kernel_name;
  kernel_attribs.is_jit_evaluator = params_.kernel->is_evaluator;
}

void lower(Kernel *kernel) {
  auto &config = kernel->program->config;
  config.demote_dense_struct_fors = true;
  irpass::compile_to_executable(kernel->ir.get(), config, kernel,
                                kernel->autodiff_mode,
                                /*ad_use_stack=*/false, config.print_ir,
                                /*lower_global_access=*/true,
                                /*make_thread_local=*/false);
}

}  // namespace spirv
}  // namespace lang
}  // namespace taichi
