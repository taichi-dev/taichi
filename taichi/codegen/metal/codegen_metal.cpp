#include "codegen_metal.h"

#include <functional>
#include <string>

#include "taichi/runtime/metal/api.h"
#include "taichi/rhi/metal/constants.h"
#include "taichi/codegen/metal/env_config.h"
#include "taichi/runtime/metal/features.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/math/arithmetic.h"
#include "taichi/util/line_appender.h"

namespace taichi {
namespace lang {
namespace metal {
namespace {

namespace shaders {
#define TI_INSIDE_METAL_CODEGEN
#include "taichi/runtime/metal/shaders/ad_stack.metal.h"
#include "taichi/runtime/metal/shaders/helpers.metal.h"
#include "taichi/runtime/metal/shaders/init_randseeds.metal.h"
#include "taichi/runtime/metal/shaders/print.metal.h"
#include "taichi/runtime/metal/shaders/runtime_kernels.metal.h"
#undef TI_INSIDE_METAL_CODEGEN

#include "taichi/runtime/metal/shaders/print.metal.h"
#include "taichi/runtime/metal/shaders/runtime_structs.metal.h"

}  // namespace shaders

constexpr char kKernelThreadIdName[] = "utid_";        // 'u' for unsigned
constexpr char kKernelGridSizeName[] = "ugrid_size_";  // 'u' for unsigned
constexpr char kKernelTidInSimdgroupName[] = "utid_in_simdg_";
constexpr char kRootBufferName[] = "root_addr";
constexpr char kGlobalTmpsBufferName[] = "global_tmps_addr";
constexpr char kContextBufferName[] = "ctx_addr";
constexpr char kContextVarName[] = "kernel_ctx_";
constexpr char kRuntimeBufferName[] = "runtime_addr";
constexpr char kRuntimeVarName[] = "runtime_";
constexpr char kPrintAssertBufferName[] = "print_assert_addr";
constexpr char kPrintAllocVarName[] = "print_alloc_";
constexpr char kAssertRecorderVarName[] = "assert_rec_";
constexpr char kLinearLoopIndexName[] = "linear_loop_idx_";
constexpr char kElementCoordsVarName[] = "elem_coords_";
constexpr char kRandStateVarName[] = "rand_state_";
constexpr char kMemAllocVarName[] = "mem_alloc_";
constexpr char kTlsBufferName[] = "tls_buffer_";

using BufferType = BufferDescriptor::Type;
using BufferDescSet =
    std::unordered_set<BufferDescriptor, BufferDescriptor::Hasher>;

std::string ndarray_buffer_name(int arg_id) {
  return fmt::format("ndarray_addr_{}", arg_id);
}

std::string buffer_to_name(const BufferDescriptor &b) {
  switch (b.type()) {
    case BufferType::Root:
      return fmt::format("{}_{}", kRootBufferName, b.root_id());
    case BufferType::GlobalTmps:
      return kGlobalTmpsBufferName;
    case BufferType::Context:
      return kContextBufferName;
    case BufferType::Runtime:
      return kRuntimeBufferName;
    case BufferType::Print:
      return kPrintAssertBufferName;
    case BufferType::Ndarray:
      return ndarray_buffer_name(b.ndarray_arg_id());
    default:
      TI_NOT_IMPLEMENTED;
      break;
  }
  return {};
}

bool is_ret_type_bit_pointer(Stmt *s) {
  if (auto *ty = s->ret_type->cast<PointerType>()) {
    // Don't use as() directly, it would fail when we inject a global tmp.
    return ty->is_bit_pointer();
  }
  return false;
}

bool is_full_bits(int bits) {
  return bits == (sizeof(uint32_t) * 8);
}

void validate_qfxt_for_metal(QuantFixedType *qfxt) {
  if (qfxt->get_compute_type()->as<PrimitiveType>() != PrimitiveType::f32) {
    TI_ERROR("Metal only supports 32-bit float");
  }
}

class RootIdsExtractor : public BasicStmtVisitor {
 public:
  static std::unordered_set<int> run(Stmt *s) {
    RootIdsExtractor re;
    s->accept(&re);
    return re.roots_;
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->task_type == OffloadedStmt::TaskType::struct_for) {
      auto *cur = stmt->snode;
      while (cur->parent) {
        cur = cur->parent;
      }
      TI_ASSERT(cur->type == SNodeType::root);
      roots_.insert(cur->id);
    }
    BasicStmtVisitor::visit(stmt);
  }

  void visit(GetRootStmt *stmt) override {
    roots_.insert(stmt->root()->id);
  }

 private:
  using BasicStmtVisitor::visit;
  std::unordered_set<int> roots_;
};

class TaskPreprocessor final : public BasicStmtVisitor {
 public:
  struct Result {
    bool should_init_randseeds{false};
    std::unordered_map<int, int> arr_args_order;
  };

  static Result run(Stmt *s) {
    TaskPreprocessor tp;
    s->accept(&tp);
    return tp.res_;
  }

 protected:
  void visit(RandStmt *) override {
    res_.should_init_randseeds = true;
  }

  void visit(ArgLoadStmt *stmt) override {
    if (!stmt->is_ptr) {
      return;
    }
    const auto arg_id = stmt->arg_id;
    if (res_.arr_args_order.count(arg_id) > 0) {
      return;
    }
    const int order = res_.arr_args_order.size();
    res_.arr_args_order[arg_id] = order;
  }

  using BasicStmtVisitor::visit;

  TaskPreprocessor() = default;
  Result res_;
};

class KernelCodegenImpl : public IRVisitor {
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
  struct Config {
    bool allow_simdgroup = true;
  };
  // TODO(k-ye): Create a Params to hold these ctor params.
  KernelCodegenImpl(const std::string &taichi_kernel_name,
                    Kernel *kernel,
                    const CompiledRuntimeModule *compiled_runtime_module,
                    const std::vector<CompiledStructs> &compiled_snode_trees,
                    PrintStringTable *print_strtab,
                    const Config &config,
                    OffloadedStmt *offloaded)
      : mtl_kernel_prefix_(taichi_kernel_name),
        kernel_(kernel),
        compiled_runtime_module_(compiled_runtime_module),
        compiled_snode_trees_(compiled_snode_trees),
        print_strtab_(print_strtab),
        cgen_config_(config),
        offloaded_(offloaded),
        ctx_attribs_(*kernel_) {
    ti_kernel_attribs_.name = taichi_kernel_name;
    ti_kernel_attribs_.is_jit_evaluator = kernel->is_evaluator;
    for (const auto s : kAllSections) {
      section_appenders_[s] = LineAppender();
    }

    for (int i = 0; i < compiled_snode_trees_.size(); ++i) {
      const auto &cst = compiled_snode_trees_[i];
      for (const auto &[node_id, _] : cst.snode_descriptors) {
        RootInfo ri{};
        ri.snode_id = cst.root_id;
        ri.index_in_cst = i;
        snode_to_roots_[node_id] = ri;
      }
    }
  }

  CompiledKernelData run() {
    emit_headers();
    generate_structs();
    generate_kernels();

    CompiledKernelData res;
    res.kernel_name = mtl_kernel_prefix_;
    res.kernel_attribs = std::move(ti_kernel_attribs_);
    res.ctx_attribs = std::move(ctx_attribs_);

    auto &source_code = res.source_code;
    source_code += section_appenders_.at(Section::Headers).lines();
    source_code += "namespace {\n";
    source_code += section_appenders_.at(Section::Structs).lines();
    source_code += section_appenders_.at(Section::KernelFuncs).lines();
    source_code += "}  // namespace\n";
    source_code += section_appenders_.at(Section::Kernels).lines();
    return res;
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
    emit("constexpr {} {} = {};",
         metal_data_type_name(const_stmt->element_type()),
         const_stmt->raw_name(), const_stmt->val.stringify());
  }

  void visit(LocalLoadStmt *stmt) override {
    auto ptr = stmt->src;
    emit("const {} {}({});", metal_data_type_name(stmt->element_type()),
         stmt->raw_name(), ptr->raw_name());
  }

  void visit(LocalStoreStmt *stmt) override {
    emit(R"({} = {};)", stmt->dest->raw_name(), stmt->val->raw_name());
  }

  void visit(GetRootStmt *stmt) override {
    const auto root_id = stmt->root()->id;
    root_id_to_stmts_[root_id] = stmt;
    const auto &cst = get_compiled_snode_tree(stmt->root());
    const auto root_desc = BufferDescriptor::root(root_id);
    emit(R"({} {}({});)", cst.root_snode_type_name, stmt->raw_name(),
         buffer_to_name(root_desc));
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
    const auto *sn = stmt->snode;
    std::string parent;
    if (stmt->input_snode) {
      parent = stmt->input_snode->raw_name();
    } else {
      const auto *root_stmt =
          root_id_to_stmts_.at(snode_to_roots_.at(sn->id).snode_id);
      parent = root_stmt->raw_name();
    }
    const auto snty = sn->type;
    if (snty == SNodeType::bit_struct) {
      // Example *bit_struct* struct generated on Metal:
      //
      // struct Sx {
      //   // bit_struct
      //   Sx(device byte *b, ...) : base(b) {}
      //   device byte *base;
      // };
      emit("auto {} = {}.base;", stmt->raw_name(), parent);
      return;
    }
    const std::string index_name = stmt->input_index->raw_name();
    // Example SNode struct generated on Metal:
    //
    // struct S1 {
    //   // dense
    //   S1(device byte *addr, ...) { rep_.init(addr); }
    //   S1_ch children(int i) { return {rep_.addr() + (i * elem_stride)}; }
    //   inline void activate(int i) { rep_.activate(i); }
    //   ...
    //  private:
    //   SNodeRep_dense rep_;
    // };
    if (stmt->activate) {
      TI_ASSERT(is_supported_sparse_type(snty));
      emit("{}.activate({});", parent, index_name);
    }
    emit(R"({}_ch {} = {}.children({});)", sn->node_type_name, stmt->raw_name(),
         parent, index_name);
  }

  void visit(GetChStmt *stmt) override {
    auto *in_snode = stmt->input_snode;
    auto *out_snode = stmt->output_snode;
    if (in_snode->type == SNodeType::bit_struct) {
      TI_ASSERT(stmt->ret_type->as<PointerType>()->is_bit_pointer());
      const auto *bit_struct_ty = in_snode->dt->cast<BitStructType>();
      const auto bit_offset =
          bit_struct_ty->get_member_bit_offset(out_snode->id_in_bit_struct);
      // stmt->input_ptr is the "base" member in the generated SNode struct.
      emit("SNodeBitPointer {}({}, /*offset=*/{});", stmt->raw_name(),
           stmt->input_ptr->raw_name(), bit_offset);
      return;
    }
    // E.g. `parent.get*(runtime, mem_alloc)`
    const auto get_call =
        fmt::format("{}.get{}({}, {})", stmt->input_ptr->raw_name(), stmt->chid,
                    kRuntimeVarName, kMemAllocVarName);
    if (out_snode->is_place()) {
      emit(R"(device {}* {} = {}.val;)", metal_data_type_name(out_snode->dt),
           stmt->raw_name(), get_call);
    } else {
      emit(R"({} {} = {};)", out_snode->node_type_name, stmt->raw_name(),
           get_call);
    }
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
        TI_ASSERT(stmt->ret_type->is_primitive(PrimitiveTypeID::i32));
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
    if (!is_ret_type_bit_pointer(stmt->dest)) {
      emit(R"(*{} = {};)", stmt->dest->raw_name(), stmt->val->raw_name());
      return;
    }
    handle_bit_pointer_global_store(stmt);
  }

  void visit(GlobalLoadStmt *stmt) override {
    std::string rhs_expr;
    if (!is_ret_type_bit_pointer(stmt->src)) {
      rhs_expr = fmt::format("*{}", stmt->src->raw_name());
    } else {
      rhs_expr = construct_bit_pointer_global_load(stmt);
    }
    emit("const auto {} = {};", stmt->raw_name(), rhs_expr);
  }

  void visit(ArgLoadStmt *stmt) override {
    const auto dt = metal_data_type_name(stmt->element_type());
    if (stmt->is_ptr) {
      const auto type_str = fmt::format("device {} *", dt);
      emit("{}{} = reinterpret_cast<{}>({});", type_str, stmt->raw_name(),
           type_str, ndarray_buffer_name(stmt->arg_id));
    } else {
      emit("const {} {} = *{}.arg{}();", dt, stmt->raw_name(), kContextVarName,
           stmt->arg_id);
    }
  }

  void visit(ReturnStmt *stmt) override {
    // TODO: use stmt->ret_id instead of 0 as index
    int idx{0};
    for (auto &value : stmt->values) {
      emit("{}.ret0()[{}] = {};", kContextVarName, idx, value->raw_name());
      idx++;
    }
  }

  void visit(ExternalPtrStmt *stmt) override {
    // Used mostly for transferring data between host (e.g. numpy array) and
    // Metal.
    const auto linear_index_name =
        fmt::format("{}_linear_index_", stmt->raw_name());
    emit("int {} = 0;", linear_index_name);
    emit("{{");
    {
      ScopedIndent s(current_appender());
      const auto *argload = stmt->base_ptr->as<ArgLoadStmt>();
      const int arg_id = argload->arg_id;
      const int num_indices = stmt->indices.size();
      const auto &element_shape = stmt->element_shape;
      std::vector<std::string> size_exprs;
      const auto layout = stmt->element_dim <= 0 ? ExternalArrayLayout::kAOS
                                                 : ExternalArrayLayout::kSOA;
      const int arr_shape_len = num_indices - element_shape.size();
      const size_t element_shape_index_offset =
          (layout == ExternalArrayLayout::kAOS) ? arr_shape_len : 0;
      for (int i = 0; i < arr_shape_len; i++) {
        std::string var_name =
            fmt::format("{}_arr_dim{}_", stmt->raw_name(), i);
        emit("const int {} = {}.extra_arg({}, {});", var_name, kContextVarName,
             arg_id, i);
        size_exprs.push_back(std::move(var_name));
      }
      size_t size_var_index = 0;
      for (int i = 0; i < num_indices; i++) {
        if (i >= element_shape_index_offset &&
            i < element_shape_index_offset + element_shape.size()) {
          emit("{} *= {};", linear_index_name,
               element_shape[i - element_shape_index_offset]);
        } else {
          emit("{} *= {};", linear_index_name, size_exprs[size_var_index++]);
        }
        emit("{} += {};", linear_index_name, stmt->indices[i]->raw_name());
      }
      TI_ASSERT(size_var_index == arr_shape_len);
    }
    emit("}}");

    const auto dt = metal_data_type_name(stmt->element_type());
    emit("device {} *{} = ({} + {});", dt, stmt->raw_name(),
         stmt->base_ptr->raw_name(), linear_index_name);
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    const auto dt = metal_data_type_name(stmt->element_type().ptr_removed());
    emit("device {}* {} = reinterpret_cast<device {}*>({} + {});", dt,
         stmt->raw_name(), dt, kGlobalTmpsBufferName, stmt->offset);
  }

  void visit(ThreadLocalPtrStmt *stmt) override {
    emit("thread auto* {} = reinterpret_cast<thread {}*>({} + {});",
         stmt->raw_name(),
         metal_data_type_name(stmt->element_type().ptr_removed()),
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
        emit("const int {} = {}.at[{}];", stmt_name, kElementCoordsVarName,
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

  void visit(DecorationStmt *stmt) override {
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
      if (is_integral(bin->ret_type)) {
        emit("const {} {} = ifloordiv({}, {});", dt_name, bin_name, lhs_name,
             rhs_name);
      } else {
        emit("const {} {} = floor({} / {});", dt_name, bin_name, lhs_name,
             rhs_name);
      }
      return;
    }
    if (op_type == BinaryOpType::pow && is_integral(bin->ret_type)) {
      // TODO(k-ye): Make sure the type is not i64?
      emit("const {} {} = pow_i32({}, {});", dt_name, bin_name, lhs_name,
           rhs_name);
      return;
    }
    const auto binop = metal_binary_op_type_symbol(op_type);
    if (is_metal_binary_op_infix(op_type)) {
      if (is_comparison(op_type)) {
        // TODO(#577): Taichi uses -1 as true due to LLVM i1... See
        // https://github.com/taichi-dev/taichi/blob/6989c0e21d437a9ffdc0151cee9d3aa2aaa2241d/taichi/codegen/llvm/codegen_llvm.cpp#L564
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

    if (is_ret_type_bit_pointer(stmt->dest)) {
      handle_bit_pointer_atomics(stmt);
      return;
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
    const auto dt = stmt->val->element_type();
    if (dt->is_primitive(PrimitiveTypeID::i32)) {
      emit(
          "const auto {} = atomic_fetch_{}_explicit((device atomic_int*){}, "
          "{}, "
          "metal::memory_order_relaxed);",
          stmt->raw_name(), op_name, stmt->dest->raw_name(), val_var);
    } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
      emit(
          "const auto {} = atomic_fetch_{}_explicit((device atomic_uint*){}, "
          "{}, "
          "metal::memory_order_relaxed);",
          stmt->raw_name(), op_name, stmt->dest->raw_name(), val_var);
    } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
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

    const auto root_ids = RootIdsExtractor::run(stmt);
    BufferDescSet used_root_descs;
    for (const auto rid : root_ids) {
      used_root_descs.insert(BufferDescriptor::root(rid));
    }
    root_id_to_stmts_.clear();
    auto preproc_res = TaskPreprocessor::run(stmt);

    using Type = OffloadedStmt::TaskType;
    if (stmt->task_type == Type::serial) {
      // For serial tasks, there is only one thread, so different calls to
      // random() is guaranteed to produce different results.
      preproc_res.should_init_randseeds = false;
      generate_serial_kernel(stmt, used_root_descs, preproc_res);
    } else if (stmt->task_type == Type::range_for) {
      generate_range_for_kernel(stmt, used_root_descs, preproc_res);
    } else if (stmt->task_type == Type::struct_for) {
      generate_struct_for_kernel(stmt, used_root_descs, preproc_res);
    } else if (stmt->task_type == Type::listgen) {
      add_runtime_list_op_kernel(stmt);
    } else if (stmt->task_type == Type::gc) {
      add_gc_op_kernels(stmt);
    } else {
      TI_ERROR("Unsupported offload type={} on Metal arch", stmt->task_name());
    }
    is_top_level_ = true;
  }

  void visit(ClearListStmt *stmt) override {
    // TODO: Try to move this into shaders/runtime_utils.metal.h
    const std::string listmgr = fmt::format("listmgr_{}", stmt->raw_name());
    emit("ListManager {};", listmgr);
    emit("{}.lm_data = ({}->snode_lists + {});", listmgr, kRuntimeVarName,
         stmt->snode->id);
    emit("{}.clear();", listmgr);
    used_features()->sparse = true;
  }

  void visit(WhileControlStmt *stmt) override {
    emit("if (!{}) break;", stmt->cond->raw_name());
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
         data_type_name(stmt->ret_type), kRandStateVarName);
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
          TI_ASSERT_INFO(
              dt->is_primitive(PrimitiveTypeID::i32) ||
                  dt->is_primitive(PrimitiveTypeID::u32) ||
                  dt->is_primitive(PrimitiveTypeID::f32),
              "print() only supports i32, u32 or f32 scalars for now.");
          emit("{}.pm_set_{}({}, {});", msg_var_name, data_type_name(dt), i,
               arg_stmt->raw_name());
        } else {
          const int str_id = print_strtab_->put(std::get<std::string>(entry));
          emit("{}.pm_set_str({}, {});", msg_var_name, i, str_id);
        }
      }
    }
    emit("}}");
  }

  void visit(AssertStmt *stmt) override {
    used_features()->assertion = true;

    const auto &args = stmt->args;
    // +1 because the assertion message template itself takes one slot
    const auto num_args = args.size() + 1;
    TI_ASSERT_INFO(num_args <= shaders::kMetalMaxNumAssertArgs,
                   "[Metal] Too many args in assert()");
    emit("if (!({})) {{", stmt->cond->raw_name());
    {
      ScopedIndent s(current_appender());
      // Only record the message for the first-time assertion failure.
      emit("if ({}.mark_first_failure()) {{", kAssertRecorderVarName);
      {
        ScopedIndent s2(current_appender());
        emit("{}.set_num_args({});", kAssertRecorderVarName, num_args);
        const std::string asst_var_name = stmt->raw_name() + "_msg_";
        emit("PrintMsg {}({}.msg_buf_addr(), {});", asst_var_name,
             kAssertRecorderVarName, num_args);
        const int msg_str_id = print_strtab_->put(stmt->text);
        emit("{}.pm_set_str(/*i=*/0, {});", asst_var_name, msg_str_id);
        for (int i = 1; i < num_args; ++i) {
          auto *arg = args[i - 1];
          const auto ty = arg->element_type();
          if (ty->is_primitive(PrimitiveTypeID::i32) ||
              ty->is_primitive(PrimitiveTypeID::f32)) {
            emit("{}.pm_set_{}({}, {});", asst_var_name, data_type_name(ty), i,
                 arg->raw_name());
          } else {
            TI_ERROR(
                "[Metal] assert() only supports i32 or f32 scalars for now.");
          }
        }
      }
      emit("}}");
      // This has failed, no point executing the rest of the kernel.
      emit("return;");
    }
    emit("}}");
  }

  void visit(AdStackAllocaStmt *stmt) override {
    TI_ASSERT_INFO(
        stmt->max_size > 0,
        "Adaptive autodiff stack's size should have been determined.");

    const auto &var_name = stmt->raw_name();
    emit("byte {}[{}];", var_name, stmt->size_in_bytes());
    emit("mtl_ad_stack_init({});", var_name);
  }

  void visit(AdStackPopStmt *stmt) override {
    emit("mtl_ad_stack_pop({});", stmt->stack->raw_name());
  }

  void visit(AdStackPushStmt *stmt) override {
    auto *stack = stmt->stack->as<AdStackAllocaStmt>();
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

  void visit(AdStackLoadTopStmt *stmt) override {
    auto *stack = stmt->stack->as<AdStackAllocaStmt>();
    const auto primal_name = stmt->raw_name() + "_primal_";
    emit(
        "thread auto* {} = reinterpret_cast<thread "
        "{}*>(mtl_ad_stack_top_primal({}, {}));",
        primal_name, metal_data_type_name(stmt->element_type()),
        stack->raw_name(), stack->element_size_in_bytes());
    emit("const auto {} = *{};", stmt->raw_name(), primal_name);
  }

  void visit(AdStackLoadTopAdjStmt *stmt) override {
    auto *stack = stmt->stack->as<AdStackAllocaStmt>();
    const auto adjoint_name = stmt->raw_name() + "_adjoint_";
    emit(
        "thread auto* {} = reinterpret_cast<thread "
        "{}*>(mtl_ad_stack_top_adjoint({}, {}));",
        adjoint_name, metal_data_type_name(stmt->element_type()),
        stack->raw_name(), stack->element_size_in_bytes());
    emit("const auto {} = *{};", stmt->raw_name(), adjoint_name);
  }

  void visit(AdStackAccAdjointStmt *stmt) override {
    auto *stack = stmt->stack->as<AdStackAllocaStmt>();
    const auto adjoint_name = stmt->raw_name() + "_adjoint_";
    emit(
        "thread auto* {} = reinterpret_cast<thread "
        "{}*>(mtl_ad_stack_top_adjoint({}, {}));",
        adjoint_name, metal_data_type_name(stmt->element_type()),
        stack->raw_name(), stack->element_size_in_bytes());
    emit("*{} += {};", adjoint_name, stmt->v->raw_name());
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    const auto name = stmt->raw_name();
    const auto arg_id = stmt->arg_id;
    const auto axis = stmt->axis;
    emit("const int {} = {}.extra_arg({}, {});", name, kContextVarName, arg_id,
         axis);
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
    current_appender().append_raw(
        compiled_runtime_module_->runtime_utils_source_code);
    emit("");
    for (const auto &cst : compiled_snode_trees_) {
      current_appender().append_raw(cst.snode_structs_source_code);
      emit("");
    }

    current_appender().append_raw(shaders::kMetalAdStackSourceCode);
    emit("");
    current_appender().append_raw(shaders::kMetalPrintSourceCode);
    emit("");
    emit_kernel_args_struct();
    emit("");
    current_appender().append_raw(shaders::kMetalInitRandseedsSourceCode);
    emit("");
  }

  void handle_bit_pointer_global_store(GlobalStoreStmt *stmt) {
    auto *ptr_type = stmt->dest->ret_type->as<PointerType>();
    TI_ASSERT(ptr_type->is_bit_pointer());
    auto *pointee_type = ptr_type->get_pointee_type();
    QuantIntType *qit = nullptr;
    std::string store_value_expr;
    if (auto *qit_cast = pointee_type->cast<QuantIntType>()) {
      qit = qit_cast;
      store_value_expr = stmt->val->raw_name();
    } else if (auto *qfxt = pointee_type->cast<QuantFixedType>()) {
      validate_qfxt_for_metal(qfxt);
      auto *digits_qit = qfxt->get_digits_type()->as<QuantIntType>();
      qit = digits_qit;
      store_value_expr = construct_quant_fixed_to_quant_int_expr(
          stmt->val, qfxt->get_scale(), digits_qit);
    } else {
      TI_NOT_IMPLEMENTED;
    }
    // Type of |stmt->dest| is SNodeBitPointer
    const auto num_bits = qit->get_num_bits();
    if (is_full_bits(num_bits)) {
      emit("mtl_set_full_bits({}, {});", stmt->dest->raw_name(),
           store_value_expr);
    } else {
      emit("mtl_set_partial_bits({},", stmt->dest->raw_name());
      emit("    {},", store_value_expr);
      emit("    /*bits=*/{});", num_bits);
    }
  }

  // Returns the expression of the load result
  std::string construct_bit_pointer_global_load(GlobalLoadStmt *stmt) const {
    auto *ptr_type = stmt->src->ret_type->as<PointerType>();
    TI_ASSERT(ptr_type->is_bit_pointer());
    auto *pointee_type = ptr_type->get_pointee_type();
    if (auto *qit = pointee_type->cast<QuantIntType>()) {
      return construct_load_quant_int(stmt->src, qit);
    } else if (auto *qfxt = pointee_type->cast<QuantFixedType>()) {
      validate_qfxt_for_metal(qfxt);
      const auto loaded = construct_load_quant_int(
          stmt->src, qfxt->get_digits_type()->as<QuantIntType>());
      // Computes `float(digits_expr) * scale`
      // See LLVM backend's reconstruct_quant_fixed()
      return fmt::format("(static_cast<float>({}) * {})", loaded,
                         qfxt->get_scale());
    }
    TI_NOT_IMPLEMENTED;
    return "";
  }

  void handle_bit_pointer_atomics(AtomicOpStmt *stmt) {
    TI_ERROR_IF(stmt->op_type != AtomicOpType::add,
                "Only atomic add is supported for bit pointer types");
    // Type of |dest_ptr| is SNodeBitPointer
    const auto *dest_ptr = stmt->dest;
    auto *ptr_type = dest_ptr->ret_type->as<PointerType>();
    TI_ASSERT(ptr_type->is_bit_pointer());
    auto *pointee_type = ptr_type->get_pointee_type();
    QuantIntType *qit = nullptr;
    std::string val_expr;
    if (auto *qit_cast = pointee_type->cast<QuantIntType>()) {
      qit = qit_cast;
      val_expr = stmt->val->raw_name();
    } else if (auto *qfxt = pointee_type->cast<QuantFixedType>()) {
      qit = qfxt->get_digits_type()->as<QuantIntType>();
      val_expr = construct_quant_fixed_to_quant_int_expr(
          stmt->val, qfxt->get_scale(), qit);
    } else {
      TI_NOT_IMPLEMENTED;
    }
    const auto num_bits = qit->get_num_bits();
    if (is_full_bits(num_bits)) {
      emit("const auto {} = mtl_atomic_add_full_bits({}, {});",
           stmt->raw_name(), dest_ptr->raw_name(), val_expr);
    } else {
      emit("const auto {} = mtl_atomic_add_partial_bits({},", stmt->raw_name(),
           dest_ptr->raw_name());
      emit("    {},", val_expr);
      emit("    /*bits=*/{});", num_bits);
    }
  }

  // Returns the expression of `int(val_stmt * (1.0f / scale) + 0.5f)`
  std::string construct_quant_fixed_to_quant_int_expr(
      const Stmt *val_stmt,
      float64 scale,
      QuantIntType *digits_qit) const {
    DataType compute_dt(digits_qit->get_compute_type()->as<PrimitiveType>());
    // This implicitly casts double to float on the host.
    const float inv_scale = 1.0 / scale;
    // Creating an expression (instead of holding intermediate results with
    // variables) because |val_stmt| could be used multiple times. If the
    // intermediate variables are named based on |val_stmt|, it would result in
    // symbol redefinitions.
    return fmt::format(
        "mtl_quant_fixed_to_quant_int<{}>(/*inv_scale=*/{} * {})",
        metal_data_type_name(compute_dt), inv_scale, val_stmt->raw_name());
  }

  // Returns expression of the loaded integer.
  std::string construct_load_quant_int(const Stmt *bit_ptr_stmt,
                                       QuantIntType *qit) const {
    DataType compute_dt(qit->get_compute_type()->as<PrimitiveType>());
    const auto num_bits = qit->get_num_bits();
    if (is_full_bits(num_bits)) {
      return fmt::format("mtl_get_full_bits<{}>({})",
                         metal_data_type_name(compute_dt),
                         bit_ptr_stmt->raw_name());
    }
    return fmt::format("mtl_get_partial_bits<{}>({}, {})",
                       metal_data_type_name(compute_dt),
                       bit_ptr_stmt->raw_name(), num_bits);
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
        if (arg.is_array) {
          continue;
        }
        const auto dt_name = metal_data_type_name(arg.dt);
        emit("device {}* arg{}() {{", dt_name, arg.index);
        emit("  // scalar, size={} B", arg.stride);
        emit("  return (device {}*)(addr_ + {});", dt_name, arg.offset_in_mem);
        emit("}}");
      }
      for (const auto &ret : ctx_attribs_.rets()) {
        // TODO: Why return still needs this?
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
    IRNode *ast = offloaded_ ? offloaded_ : kernel_->ir.get();
    ast->accept(this);

    if (used_features()->sparse) {
      emit("");
      current_appender().append_raw(shaders::kMetalRuntimeKernelsSourceCode);
    }
  }

  std::vector<BufferDescriptor> get_used_buffer_descriptors(
      const BufferDescSet &root_buffer_descs) const {
    std::vector<BufferDescriptor> result;
    result.insert(result.end(), root_buffer_descs.begin(),
                  root_buffer_descs.end());

    std::sort(
        result.begin(), result.end(),
        [](const BufferDescriptor &lhs, const BufferDescriptor &rhs) -> bool {
          TI_ASSERT(lhs.type() == BufferType::Root);
          TI_ASSERT(rhs.type() == BufferType::Root);
          return lhs.root_id() < rhs.root_id();
        });
    result.push_back(BufferDescriptor::global_tmps());
    if (!ctx_attribs_.empty()) {
      result.push_back(BufferDescriptor::context());
    }
    result.push_back(BufferDescriptor::runtime());
    // TODO(k-ye): Bind this buffer only when print() is used.
    result.push_back(BufferDescriptor::print());
    return result;
  }

  static std::unordered_map<int, int> make_arr_args_to_binding_indices(
      const std::unordered_map<int, int> &arr_args_order,
      int binding_idx_offset) {
    auto res = arr_args_order;
    for (auto itr = res.begin(); itr != res.end(); ++itr) {
      itr->second += binding_idx_offset;
    }
    return res;
  }

  static void append_arr_buffer_descriptors(
      const std::unordered_map<int, int> &arr_bindings,
      std::vector<BufferDescriptor> *descs) {
    for (const auto &[arr_id, _] : arr_bindings) {
      descs->push_back(BufferDescriptor::ndarray(arr_id));
    }
  }

  void generate_serial_kernel(OffloadedStmt *stmt,
                              const BufferDescSet &root_buffer_descs,
                              const TaskPreprocessor::Result &preproc_res) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::serial);
    const std::string mtl_kernel_name = make_kernel_name();
    KernelAttributes ka;
    ka.name = mtl_kernel_name;
    ka.task_type = stmt->task_type;
    ka.buffers = get_used_buffer_descriptors(root_buffer_descs);
    ka.arr_args_to_binding_indices = make_arr_args_to_binding_indices(
        preproc_res.arr_args_order, ka.buffers.size());
    append_arr_buffer_descriptors(ka.arr_args_to_binding_indices,
                                  &(ka.buffers));
    ka.advisory_total_num_threads = 1;
    ka.advisory_num_threads_per_group = 1;

    emit_mtl_kernel_sig(mtl_kernel_name, ka.buffers);
    {
      ScopedIndent s(current_appender());
      emit("// serial");
      emit("if ({} > 0) return;", kKernelThreadIdName);

      current_kernel_attribs_ = &ka;
      const auto mtl_func_name = mtl_kernel_func_name(mtl_kernel_name);
      emit_mtl_kernel_func_def(mtl_func_name, ka.buffers, preproc_res,
                               stmt->body.get());
      emit_call_mtl_kernel_func(mtl_func_name, ka.buffers,
                                /*loop_index_expr=*/"0");
    }
    // Close kernel
    emit("}}\n");

    current_kernel_attribs_ = nullptr;
    mtl_kernels_attribs()->push_back(ka);
  }

  void generate_range_for_kernel(OffloadedStmt *stmt,
                                 const BufferDescSet &root_buffer_descs,
                                 const TaskPreprocessor::Result &preproc_res) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::range_for);
    const std::string mtl_kernel_name = make_kernel_name();
    KernelAttributes ka;
    ka.name = mtl_kernel_name;
    ka.task_type = stmt->task_type;
    ka.buffers = get_used_buffer_descriptors(root_buffer_descs);
    ka.arr_args_to_binding_indices = make_arr_args_to_binding_indices(
        preproc_res.arr_args_order, ka.buffers.size());
    append_arr_buffer_descriptors(ka.arr_args_to_binding_indices,
                                  &(ka.buffers));
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
      ka.advisory_total_num_threads = num_elems;
    } else {
      emit("// range_for, range known at runtime");
      begin_expr = stmt->const_begin
                       ? std::to_string(stmt->begin_value)
                       : inject_load_global_tmp(stmt->begin_offset);
      const auto end_expr = stmt->const_end
                                ? std::to_string(stmt->end_value)
                                : inject_load_global_tmp(stmt->end_offset);
      emit("const int {} = {} - {};", total_elems_name, end_expr, begin_expr);
      ka.advisory_total_num_threads = kMaxNumThreadsGridStrideLoop;
    }
    // TODO: I've seen cases where |block_dim| was set to 1...
    ka.advisory_num_threads_per_group = stmt->block_dim;
    // begin_ = thread_id   + begin_expr
    emit("const int begin_ = {} + {};", kKernelThreadIdName, begin_expr);
    // end_   = total_elems + begin_expr
    emit("const int end_ = {} + {};", total_elems_name, begin_expr);

    emit_runtime_and_memalloc_def();
    if (used_tls) {
      generate_tls_prologue(stmt);
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
                               preproc_res, stmt->body.get());
      emit_call_mtl_kernel_func(mtl_func_name, ka.buffers, extra_args,
                                /*loop_index_expr=*/"ii");
    }
    emit("}}");  // closes for loop

    if (used_tls) {
      generate_tls_epilogue(stmt);
    }

    current_appender().pop_indent();
    // Close kernel
    emit("}}\n");

    current_kernel_attribs_ = nullptr;
    mtl_kernels_attribs()->push_back(ka);
  }

  void generate_struct_for_kernel(OffloadedStmt *stmt,
                                  const BufferDescSet &root_buffer_descs,
                                  const TaskPreprocessor::Result &preproc_res) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::struct_for);
    const std::string mtl_kernel_name = make_kernel_name();

    KernelAttributes ka;
    ka.name = mtl_kernel_name;
    ka.task_type = stmt->task_type;
    ka.buffers = get_used_buffer_descriptors(root_buffer_descs);
    ka.arr_args_to_binding_indices = make_arr_args_to_binding_indices(
        preproc_res.arr_args_order, ka.buffers.size());
    append_arr_buffer_descriptors(ka.arr_args_to_binding_indices,
                                  &(ka.buffers));
    const bool used_tls = (stmt->tls_prologue != nullptr);
    KernelSigExtensions kernel_exts;
    kernel_exts.use_simdgroup = (used_tls && cgen_config_.allow_simdgroup);
    used_features()->simdgroup =
        used_features()->simdgroup || kernel_exts.use_simdgroup;
    emit_mtl_kernel_sig(mtl_kernel_name, ka.buffers, kernel_exts);

    const int sn_id = stmt->snode->id;
    // struct_for kernels use grid-stride loops
    const int total_num_elems_from_root = get_compiled_snode_tree(stmt->snode)
                                              .snode_descriptors.at(sn_id)
                                              .total_num_elems_from_root;
    ka.advisory_total_num_threads =
        std::min(total_num_elems_from_root, kMaxNumThreadsGridStrideLoop);
    ka.advisory_num_threads_per_group = stmt->block_dim;

    current_appender().push_indent();
    emit("// struct_for");
    emit_runtime_and_memalloc_def();

    if (used_tls) {
      generate_tls_prologue(stmt);
    }

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
    emit("for (int ii = {};; ii += {}) {{", kKernelThreadIdName,
         kKernelGridSizeName);
    {
      const auto belonged_root_id = snode_to_roots_.at(sn_id).snode_id;
      const auto root_desc = BufferDescriptor::root(belonged_root_id);
      ScopedIndent s2(current_appender());
      emit("const int parent_idx_ = (ii / child_num_slots);");
      emit("if (parent_idx_ >= parent_list.num_active()) break;");
      emit("const int child_idx_ = (ii % child_num_slots);");
      emit(
          "const auto parent_elem_ = "
          "parent_list.get<ListgenElement>(parent_idx_);");
      emit(
          "device auto *parent_addr_ = mtl_lgen_snode_addr(parent_elem_, {}, "
          "{}, {});",
          buffer_to_name(root_desc), kRuntimeVarName, kMemAllocVarName);
      emit("if (!is_active(parent_addr_, parent_meta, child_idx_)) continue;");
      emit("ElementCoords {};", kElementCoordsVarName);
      emit(
          "refine_coordinates(parent_elem_.coords, {}->snode_extractors[{}], "
          "child_idx_, &{});",
          kRuntimeVarName, sn_id, kElementCoordsVarName);

      current_kernel_attribs_ = &ka;
      const auto mtl_func_name = mtl_kernel_func_name(mtl_kernel_name);
      std::vector<FuncParamLiteral> extra_func_params = {
          {"thread const ElementCoords &", kElementCoordsVarName},
      };
      std::vector<std::string> extra_args = {
          kElementCoordsVarName,
      };
      if (used_tls) {
        extra_func_params.push_back({"thread char*", kTlsBufferName});
        extra_args.push_back(kTlsBufferName);
      }
      emit_mtl_kernel_func_def(mtl_func_name, ka.buffers, extra_func_params,
                               preproc_res, stmt->body.get());
      emit_call_mtl_kernel_func(mtl_func_name, ka.buffers, extra_args,
                                /*loop_index_expr=*/"ii");
    }
    emit("}}");  // closes for loop
    if (used_tls) {
      generate_tls_epilogue(stmt);
    }

    current_appender().pop_indent();
    emit("}}\n");  // closes kernel

    current_kernel_attribs_ = nullptr;
    mtl_kernels_attribs()->push_back(ka);
  }

  void generate_tls_prologue(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->tls_prologue != nullptr);
    emit("// TLS prologue");
    const std::string tls_bufi32_name = "tls_bufi32_";
    // Using |int32_t| because it aligns to 4bytes.
    emit("int32_t {}[{}];", tls_bufi32_name, (stmt->tls_size + 3) / 4);
    emit("thread char* {} = reinterpret_cast<thread char*>({});",
         kTlsBufferName, tls_bufi32_name);
    stmt->tls_prologue->accept(this);
  }

  void generate_tls_epilogue(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->tls_epilogue != nullptr);
    inside_tls_epilogue_ = true;
    emit("{{  // TLS epilogue");
    stmt->tls_epilogue->accept(this);
    inside_tls_epilogue_ = false;
    emit("}}");
  }

  void add_runtime_list_op_kernel(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->task_type == OffloadedTaskType::listgen);
    auto *const sn = stmt->snode;
    KernelAttributes ka;
    ka.name = "element_listgen";
    ka.task_type = stmt->task_type;
    // listgen kernels use grid-stride loops
    const auto &sn_descs = get_compiled_snode_tree(sn).snode_descriptors;
    ka.advisory_total_num_threads =
        std::min(total_num_self_from_root(sn_descs, sn->id),
                 kMaxNumThreadsGridStrideLoop);
    ka.advisory_num_threads_per_group = stmt->block_dim;
    ka.buffers = {BufferDescriptor::runtime(),
                  BufferDescriptor::root(snode_to_roots_.at(sn->id).snode_id),
                  BufferDescriptor::context()};

    ka.runtime_list_op_attribs = KernelAttributes::RuntimeListOpAttributes();
    ka.runtime_list_op_attribs->snode = sn;
    current_kernel_attribs_ = nullptr;

    mtl_kernels_attribs()->push_back(ka);
    used_features()->sparse = true;
  }

  void add_gc_op_kernels(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->task_type == OffloadedTaskType::gc);

    auto *const sn = stmt->snode;
    const auto &sn_descs = get_compiled_snode_tree(sn).snode_descriptors;
    // common attributes shared among the 3-stage GC kernels
    KernelAttributes ka;
    ka.task_type = OffloadedTaskType::gc;
    ka.gc_op_attribs = KernelAttributes::GcOpAttributes();
    ka.gc_op_attribs->snode = sn;
    ka.buffers = {BufferDescriptor::runtime(), BufferDescriptor::context()};
    current_kernel_attribs_ = nullptr;
    // stage 1 specific
    ka.name = "gc_compact_free_list";
    ka.advisory_total_num_threads =
        std::min(total_num_self_from_root(sn_descs, sn->id),
                 kMaxNumThreadsGridStrideLoop);
    ka.advisory_num_threads_per_group = stmt->block_dim;
    mtl_kernels_attribs()->push_back(ka);
    // stage 2 specific
    ka.name = "gc_reset_free_list";
    ka.advisory_total_num_threads = 1;
    ka.advisory_num_threads_per_group = 1;
    mtl_kernels_attribs()->push_back(ka);
    // stage 3 specific
    ka.name = "gc_move_recycled_to_free";
    ka.advisory_total_num_threads =
        std::min(total_num_self_from_root(sn_descs, sn->id),
                 kMaxNumThreadsGridStrideLoop);
    ka.advisory_num_threads_per_group = stmt->block_dim;
    mtl_kernels_attribs()->push_back(ka);

    used_features()->sparse = true;
  }

  const CompiledStructs &get_compiled_snode_tree(const SNode *sn) const {
    const auto &ri = snode_to_roots_.at(sn->id);
    return compiled_snode_trees_[ri.index_in_cst];
  }

  std::string inject_load_global_tmp(int offset,
                                     DataType dt = PrimitiveType::i32) {
    auto gtmp = Stmt::make<GlobalTemporaryStmt>(offset, dt);
    gtmp->accept(this);
    auto gload = Stmt::make<GlobalLoadStmt>(gtmp.get());
    gload->ret_type = dt;
    gload->accept(this);
    return gload->raw_name();
  }

  struct FuncParamLiteral {
    std::string type;
    std::string name;
  };

  void emit_mtl_kernel_func_def(
      const std::string &kernel_func_name,
      const std::vector<BufferDescriptor> &buffers,
      const std::vector<FuncParamLiteral> &extra_params,
      const TaskPreprocessor::Result &preproc_res,
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
      if (preproc_res.should_init_randseeds) {
        emit("mtl_init_random_seeds(({}->rand_seeds), {}, {});",
             kRuntimeVarName, kLinearLoopIndexName, kNumRandSeeds);
      }
      // Init AssertRecorder.
      emit("AssertRecorder {}({});", kAssertRecorderVarName,
           kPrintAssertBufferName);
      // Init PrintMsgAllocator.
      // The print buffer comes after (AssertRecorder + assert message buffer),
      // therefore we skip by +|kMetalAssertBufferSize|.
      emit(
          "device auto* {} = reinterpret_cast<device PrintMsgAllocator*>({} + "
          "{});",
          kPrintAllocVarName, kPrintAssertBufferName,
          shaders::kMetalAssertBufferSize);
    }
    // We do not need additional indentation, because |func_ir| itself is a
    // block, which will be indented automatically.
    func_ir->accept(this);

    emit("}}\n");
  }

  inline void emit_mtl_kernel_func_def(
      const std::string &kernel_func_name,
      const std::vector<BufferDescriptor> &buffers,
      const TaskPreprocessor::Result &preproc_res,
      Block *func_ir) {
    emit_mtl_kernel_func_def(kernel_func_name, buffers, /*extra_params=*/{},
                             preproc_res, func_ir);
  }

  void emit_call_mtl_kernel_func(const std::string &kernel_func_name,
                                 const std::vector<BufferDescriptor> &buffers,
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
      const std::vector<BufferDescriptor> &buffers,
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

  void emit_mtl_kernel_sig(const std::string &kernel_name,
                           const std::vector<BufferDescriptor> &buffers,
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
    SectionGuard(KernelCodegenImpl *kg, Section new_sec)
        : kg_(kg), saved_(kg->code_section_) {
      kg->code_section_ = new_sec;
    }

    ~SectionGuard() {
      kg_->code_section_ = saved_;
    }

   private:
    KernelCodegenImpl *const kg_;
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
  void emit(std::string f, Args &&...args) {
    current_appender().append(std::move(f), std::forward<Args>(args)...);
  }

  std::vector<KernelAttributes> *mtl_kernels_attribs() {
    return &(ti_kernel_attribs_.mtl_kernels_attribs);
  }

  TaichiKernelAttributes::UsedFeatures *used_features() {
    return &(ti_kernel_attribs_.used_features);
  }

  const std::string mtl_kernel_prefix_;
  Kernel *const kernel_;
  const CompiledRuntimeModule *const compiled_runtime_module_;
  const std::vector<CompiledStructs> &compiled_snode_trees_;
  // const bool needs_root_buffer_;
  struct RootInfo {
    int snode_id{-1};
    int index_in_cst{-1};
  };
  std::unordered_map<int, RootInfo> snode_to_roots_;
  std::unordered_map<int, const GetRootStmt *> root_id_to_stmts_;
  PrintStringTable *const print_strtab_;
  const Config &cgen_config_;
  OffloadedStmt *const offloaded_;

  TaichiKernelAttributes ti_kernel_attribs_;
  KernelContextAttributes ctx_attribs_;

  bool is_top_level_{true};
  int mtl_kernel_count_{0};
  KernelAttributes *current_kernel_attribs_{nullptr};
  bool inside_tls_epilogue_{false};
  Section code_section_{Section::Structs};
  std::unordered_map<Section, LineAppender> section_appenders_;
};

}  // namespace

CompiledKernelData run_codegen(
    const CompiledRuntimeModule *compiled_runtime_module,
    const std::vector<CompiledStructs> &compiled_snode_trees,
    Kernel *kernel,
    PrintStringTable *strtab,
    OffloadedStmt *offloaded) {
  const auto id = Program::get_kernel_id();
  const auto taichi_kernel_name(
      fmt::format("mtl_k{:04d}_{}", id, kernel->name));

  KernelCodegenImpl::Config cgen_config;
  cgen_config.allow_simdgroup = EnvConfig::instance().is_simdgroup_enabled();

  KernelCodegenImpl codegen(taichi_kernel_name, kernel, compiled_runtime_module,
                            compiled_snode_trees, strtab, cgen_config,
                            offloaded);

  return codegen.run();
}

FunctionType compile_to_metal_executable(
    Kernel *kernel,
    KernelManager *kernel_mgr,
    const CompiledRuntimeModule *compiled_runtime_module,
    const std::vector<CompiledStructs> &compiled_snode_trees,
    OffloadedStmt *offloaded) {
  const auto compiled_res =
      run_codegen(compiled_runtime_module, compiled_snode_trees, kernel,
                  kernel_mgr->print_strtable(), offloaded);
  kernel_mgr->register_taichi_kernel(
      compiled_res.kernel_name, compiled_res.source_code,
      compiled_res.kernel_attribs, compiled_res.ctx_attribs, kernel);
  return [kernel_mgr,
          kernel_name = compiled_res.kernel_name](RuntimeContext &ctx) {
    kernel_mgr->launch_taichi_kernel(kernel_name, &ctx);
  };
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
