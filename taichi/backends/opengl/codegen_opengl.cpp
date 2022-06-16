//#define _GLSL_DEBUG 1
#include "codegen_opengl.h"

#include <string>

#include "taichi/runtime/opengl/opengl_api.h"
#include "taichi/backends/opengl/opengl_data_types.h"
#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/util/line_appender.h"
#include "taichi/util/macros.h"

TLANG_NAMESPACE_BEGIN
namespace opengl {

namespace {

namespace shaders {
#define FOREACH_ARR_NAME(_) \
  _(arr0)                   \
  _(arr1)                   \
  _(arr2)                   \
  _(arr3)                   \
  _(arr4)                   \
  _(arr5)                   \
  _(arr6)                   \
  _(arr7)

#define TI_INSIDE_OPENGL_CODEGEN
#include "taichi/backends/opengl/shaders/atomics_macro_f32.glsl.h"
#include "taichi/runtime/opengl/shaders/runtime.h"
#include "taichi/backends/opengl/shaders/random.glsl.h"
#include "taichi/backends/opengl/shaders/fast_pow.glsl.h"
#include "taichi/backends/opengl/shaders/print.glsl.h"
#include "taichi/backends/opengl/shaders/reduction.glsl.h"

GENERATE_OPENGL_ATOMIC_F32(data);
GENERATE_OPENGL_ATOMIC_F32(gtmp);

FOREACH_ARR_NAME(GENERATE_OPENGL_ATOMIC_F32);

GENERATE_OPENGL_REDUCTION_FUNCTIONS(add, float);
GENERATE_OPENGL_REDUCTION_FUNCTIONS(max, float);
GENERATE_OPENGL_REDUCTION_FUNCTIONS(min, float);
GENERATE_OPENGL_REDUCTION_FUNCTIONS(add, int);
GENERATE_OPENGL_REDUCTION_FUNCTIONS(max, int);
GENERATE_OPENGL_REDUCTION_FUNCTIONS(min, int);
GENERATE_OPENGL_REDUCTION_FUNCTIONS(add, uint);
GENERATE_OPENGL_REDUCTION_FUNCTIONS(max, uint);
GENERATE_OPENGL_REDUCTION_FUNCTIONS(min, uint);

#undef TI_INSIDE_OPENGL_CODEGEN
#undef FOREACH_ARR_NAME
}  // namespace shaders

using irpass::ExternalPtrAccess;

std::string opengl_atomic_op_type_cap_name(AtomicOpType type) {
  static std::map<AtomicOpType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i, s) type_names[AtomicOpType::i] = "atomic" #s;
    REGISTER_TYPE(add, Add);
    REGISTER_TYPE(sub, Sub);
    // REGISTER_TYPE(mul, Mul);
    // REGISTER_TYPE(div, Div);
    REGISTER_TYPE(max, Max);
    REGISTER_TYPE(min, Min);
    REGISTER_TYPE(bit_and, And);
    REGISTER_TYPE(bit_or, Or);
    REGISTER_TYPE(bit_xor, Xor);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

#if !defined(TI_PLATFORM_WINDOWS)
#include <sys/wait.h>
#endif

class KernelGen : public IRVisitor {
 public:
  KernelGen(Kernel *kernel,
            const StructCompiledResult *struct_compiled,
            const std::string &kernel_name,
            bool allows_nv_shader_ext)
      : kernel_(kernel),
        struct_compiled_(struct_compiled),
        kernel_name_(kernel_name),
        allows_nv_shader_ext_(allows_nv_shader_ext),
        root_snode_type_name_(struct_compiled->root_snode_type_name),
        glsl_kernel_prefix_(kernel_name) {
    compiled_program_.init_args(kernel);
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

 private:
  const Kernel *kernel_;
  const StructCompiledResult *struct_compiled_;
  const std::string kernel_name_;
  const bool allows_nv_shader_ext_;
  const std::string root_snode_type_name_;
  const std::string glsl_kernel_prefix_;

  GetRootStmt *root_stmt_;
  int glsl_kernel_count_{0};
  bool is_top_level_{true};
  CompiledTaichiKernel compiled_program_;
  UsedFeature used;  // TODO: is this actually per-offload?

  // per-offload variables:
  LineAppender line_appender_;
  LineAppender line_appender_header_;
  std::string glsl_kernel_name_;
  int num_workgroups_{1};
  int workgroup_size_{1};
  bool used_tls_;  // TODO: move into UsedFeature?
  std::unordered_map<int, irpass::ExternalPtrAccess> extptr_access_;
  std::unordered_set<std::string> loaded_args_;

  template <typename... Args>
  void emit(std::string f, Args &&...args) {
    line_appender_.append(std::move(f), std::move(args)...);
  }

  void generate_header() {
    emit("const float inf = 1.0f / 0.0f;");
    emit("const float nan = 0.0f / 0.0f;");
  }

  // Note that the following two functions not only returns the corresponding
  // data type, but also **records** the usage of data types to UsedFeatures.
  std::string opengl_data_type_short_name(DataType dt) {
    if (dt->is_primitive(PrimitiveTypeID::i64) ||
        dt->is_primitive(PrimitiveTypeID::u64)) {
      if (!TI_OPENGL_REQUIRE(used, GL_ARB_gpu_shader_int64)) {
        TI_ERROR(
            "Extension GL_ARB_gpu_shader_int64 not supported on your OpenGL");
      }
    }
    if (dt->is_primitive(PrimitiveTypeID::f32))
      used.float32 = true;
    if (dt->is_primitive(PrimitiveTypeID::f64))
      used.float64 = true;
    if (dt->is_primitive(PrimitiveTypeID::i32))
      used.int32 = true;
    if (dt->is_primitive(PrimitiveTypeID::i64))
      used.int64 = true;
    if (dt->is_primitive(PrimitiveTypeID::u32))
      used.uint32 = true;
    if (dt->is_primitive(PrimitiveTypeID::u64))
      used.uint64 = true;
    return data_type_name(dt);
  }

  std::string opengl_data_type_name(DataType dt) {
    return opengl::opengl_data_type_name(dt);
  }

  std::string gen_layout_line(std::string dt,
                              std::string dtype,
                              std::string buf,
                              std::string bind_id) {
    return fmt::format(
        "layout(std430, binding = {}) buffer {}_{} {{ {} _{}_{}_[];}}; \n",
        bind_id, buf, dt, dtype, buf, dt);
  }

  std::string gen_buffer_registration(const UsedFeature &used,
                                      std::string buf,
                                      std::string bind_id) {
    std::string res = "";
    if (used.int32)
      res += gen_layout_line("i32", "int", buf, bind_id);
    if (used.int64)
      res += gen_layout_line("i64", "int64_t", buf, bind_id);
    if (used.uint32)
      res += gen_layout_line("u32", "uint", buf, bind_id);
    if (used.uint64)
      res += gen_layout_line("u64", "uint64_t", buf, bind_id);
    if (used.float32)
      res += gen_layout_line("f32", "float", buf, bind_id);
    if (used.float64)
      res += gen_layout_line("f64", "double", buf, bind_id);
    return res;
  }

  void generate_task_bottom(OffloadedTaskType task_type,
                            std::string range_hint) {
    emit("void main()");
    emit("{{");
    if (used.random)
      emit("  _init_rand();");
    if (glsl_kernel_name_.size())
      emit("  {}();", glsl_kernel_name_);
    emit("}}");

    if (used.print)  // the runtime buffer is only used for print now..
      line_appender_header_.append_raw(shaders::kOpenGlRuntimeSourceCode);

    std::string kernel_header;
    if (used.buf_data)
      kernel_header += gen_buffer_registration(
          used, "data", std::to_string(static_cast<int>(GLBufId::Root)));
    if (used.buf_gtmp)
      kernel_header += gen_buffer_registration(
          used, "gtmp", std::to_string(static_cast<int>(GLBufId::Gtmp)));
    if (used.buf_args)
      kernel_header += gen_buffer_registration(
          used, "args", std::to_string(static_cast<int>(GLBufId::Args)));
    for (auto [arr_id, bind_idx] : used.arr_arg_to_bind_idx) {
      kernel_header += gen_buffer_registration(
          used, "arr" + std::to_string(arr_id), std::to_string(bind_idx));
    }

    if (used.simulated_atomic_float) {
      if (used.buf_data) {
        kernel_header += shaders::kOpenGlAtomicF32Source_data;
      }
      if (used.buf_gtmp) {
        kernel_header += shaders::kOpenGlAtomicF32Source_gtmp;
      }
      std::unordered_set<int> arr_ids;
      for ([[maybe_unused]] const auto &[arr_id, bind_idx] :
           used.arr_arg_to_bind_idx) {
        arr_ids.insert(arr_id);
      }

#define FOREACH_ARR_ID(_) \
  _(0)                    \
  _(1)                    \
  _(2)                    \
  _(3)                    \
  _(4)                    \
  _(5)                    \
  _(6)                    \
  _(7)

#define ADD_ARR_ATOMIC_F32_SOURCE(id)                         \
  if (arr_ids.count(id)) {                                    \
    kernel_header += shaders::kOpenGlAtomicF32Source_arr##id; \
  }

      FOREACH_ARR_ID(ADD_ARR_ATOMIC_F32_SOURCE);
#undef ADD_ARR_ATOMIC_F32_SOURCE
#undef FOREACH_ARR_ID
    }

    if (used.reduction) {
      line_appender_header_.append_raw(shaders::kOpenGLReductionCommon);
      kernel_header += shaders::kOpenGlReductionSource_add_float;
      kernel_header += shaders::kOpenGlReductionSource_max_float;
      kernel_header += shaders::kOpenGlReductionSource_min_float;
      kernel_header += shaders::kOpenGlReductionSource_add_int;
      kernel_header += shaders::kOpenGlReductionSource_max_int;
      kernel_header += shaders::kOpenGlReductionSource_min_int;
      kernel_header += shaders::kOpenGlReductionSource_add_uint;
      kernel_header += shaders::kOpenGlReductionSource_max_uint;
      kernel_header += shaders::kOpenGlReductionSource_min_uint;
    }

    line_appender_header_.append_raw(kernel_header);

    if (used.random) {
      line_appender_header_.append_raw(shaders::kOpenGLRandomSourceCode);
    }

    if (used.fast_pow) {
      line_appender_header_.append_raw(shaders::kOpenGLFastPowSourceCode);
    }
    if (used.print) {
      line_appender_header_.append_raw(shaders::kOpenGLPrintSourceCode);
    }

    std::string extensions = "";
#define PER_OPENGL_EXTENSION(x) \
  if (used.extension_##x)       \
    extensions += "#extension " #x ": enable\n";
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION
    auto kernel_src_code =
        (is_gles() ? "#version 310 es\n" : "#version 430 core\n") + extensions +
        "precision highp float;\n" + line_appender_header_.lines() +
        line_appender_.lines();
    auto &config = kernel_->program->config;
    const int prescribed_block_dim = config.max_block_dim;
    workgroup_size_ = prescribed_block_dim > 0
                          ? std::min(workgroup_size_, prescribed_block_dim)
                          : workgroup_size_;
    compiled_program_.add(std::move(glsl_kernel_name_), kernel_src_code,
                          task_type, range_hint, num_workgroups_,
                          workgroup_size_, &this->extptr_access_);
    if (config.print_kernel_llvm_ir) {
      static FileSequenceWriter writer("shader{:04d}.comp",
                                       "OpenGL compute shader");
      writer.write(kernel_src_code);
    }
    line_appender_header_.clear_all();
    line_appender_.clear_all();
    num_workgroups_ = 1;
    workgroup_size_ = 1;
  }

  void visit(Block *stmt) override {
    if (!is_top_level_)
      line_appender_.push_indent();
    for (auto &s : stmt->statements) {
      s->accept(this);
    }
    if (!is_top_level_)
      line_appender_.pop_indent();
  }

  void visit(Stmt *stmt) override {
    TI_ERROR("[glsl] unsupported statement type {}", typeid(*stmt).name());
  }

  void visit(PrintStmt *stmt) override {
    used.print = true;

    int size = stmt->contents.size();
    if (size > MAX_CONTENTS_PER_MSG) {
      TI_WARN("[glsl] printing too much contents: {} > {}, clipping", size,
              MAX_CONTENTS_PER_MSG);
    }
    auto msgid_name = fmt::format("_mi_{}", stmt->short_name());
    emit("int {} = atomicAdd(_msg_count_, 1);", msgid_name);
    emit("{} %= {};", msgid_name, MAX_MESSAGES);
    for (int i = 0; i < size; i++) {
      auto const &content = stmt->contents[i];

      if (std::holds_alternative<Stmt *>(content)) {
        auto arg_stmt = std::get<Stmt *>(content);
        emit("_msg_set_{}({}, {}, {});",
             opengl_data_type_short_name(arg_stmt->ret_type), msgid_name, i,
             arg_stmt->short_name());

      } else {
        auto str = std::get<std::string>(content);
        int stridx = compiled_program_.lookup_or_add_string(str);
        emit("_msg_set_str({}, {}, {});", msgid_name, i, stridx);
      }
    }
    emit("_msg_set_end({}, {});", msgid_name, size);
  }

  void visit(RandStmt *stmt) override {
    used.random = true;
    // since random generator uses _gtmp_i32_ as rand state:
    used.buf_gtmp = true;
    used.int32 = true;
    emit("{} {} = _rand_{}();", opengl_data_type_name(stmt->ret_type),
         stmt->short_name(), opengl_data_type_short_name(stmt->ret_type));
  }

  void visit(LinearizeStmt *stmt) override {
    std::string val = "0";
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      val = fmt::format("({} * {} + {})", val, stmt->strides[i],
                        stmt->inputs[i]->short_name());
    }
    emit("int {} = {};", stmt->short_name(), val);
  }

  void visit(BitExtractStmt *stmt) override {
    TI_WARN(
        "BitExtractStmt visited. It should have been taken care of by the "
        "demote_operations pass.");
    emit("int {} = (({} >> {}) & ((1 << {}) - 1));", stmt->short_name(),
         stmt->input->short_name(), stmt->bit_begin,
         stmt->bit_end - stmt->bit_begin);
  }

  void visit(GetRootStmt *stmt) override {
    // Should we assert |root_stmt_| is assigned only once?
    root_stmt_ = stmt;
    emit("int {} = 0;", stmt->short_name());
  }

  void visit(SNodeLookupStmt *stmt) override {
    Stmt *parent;
    std::string parent_type;
    if (stmt->input_snode) {
      parent = stmt->input_snode;
      parent_type = stmt->snode->node_type_name;
    } else {
      TI_ASSERT(root_stmt_ != nullptr);
      parent = root_stmt_;
      parent_type = root_snode_type_name_;
    }

    emit("int {} = {} + {} * {}; // {}", stmt->short_name(),
         parent->short_name(),
         struct_compiled_->snode_map.at(parent_type).elem_stride,
         stmt->input_index->short_name(), stmt->snode->node_type_name);

    if (stmt->activate) {
      TI_ASSERT(stmt->snode->type == SNodeType::dense);
    }
  }

  void visit(AssertStmt *stmt) override {
    // TODO: do the actual assert
    TI_WARN("Assert is not supported for OpenGL arch");
  }

  void visit(SNodeOpStmt *stmt) override {  // IAPR?
    if (stmt->op_type == SNodeOpType::activate) {
      if (stmt->snode->type == SNodeType::dense ||
          stmt->snode->type == SNodeType::root) {
        // do nothing
      } else {
        TI_NOT_IMPLEMENTED
      }

    } else if (stmt->op_type == SNodeOpType::deactivate) {
      if (stmt->snode->type == SNodeType::dense ||
          stmt->snode->type == SNodeType::root) {
        // do nothing
      } else {
        TI_NOT_IMPLEMENTED
      }

    } else if (stmt->op_type == SNodeOpType::is_active) {
      TI_ASSERT(stmt->ret_type->is_primitive(PrimitiveTypeID::i32));
      if (stmt->snode->type == SNodeType::dense ||
          stmt->snode->type == SNodeType::root) {
        emit("int {} = 1;", stmt->short_name());
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  std::map<int, std::string> ptr_signats_;

  void visit(GetChStmt *stmt) override {
    used.buf_data = true;
    emit("int {} = {} + {}; // {}", stmt->short_name(),
         stmt->input_ptr->short_name(),
         struct_compiled_->snode_map.at(stmt->input_snode->node_type_name)
             .children_offsets[stmt->chid],
         stmt->output_snode->node_type_name);
    if (stmt->output_snode->is_place())
      ptr_signats_[stmt->id] = "data";
  }

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt = stmt->val->element_type();
    std::string index = stmt->dest->is<ExternalPtrStmt>()
                            ? stmt->dest->short_name()
                            : fmt::format("{} >> {}", stmt->dest->short_name(),
                                          opengl_data_address_shifter(dt));

    emit(
        "_{}_{}_[{}] = {};",
        ptr_signats_.at(stmt->dest->id),  // throw out_of_range if not a pointer
        opengl_data_type_short_name(dt), index, stmt->val->short_name());
  }

  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt = stmt->element_type();
    std::string index = stmt->src->is<ExternalPtrStmt>()
                            ? stmt->src->short_name()
                            : fmt::format("{} >> {}", stmt->src->short_name(),
                                          opengl_data_address_shifter(dt));

    emit("{} {} = _{}_{}_[{}];", opengl_data_type_name(stmt->element_type()),
         stmt->short_name(), ptr_signats_.at(stmt->src->id),
         opengl_data_type_short_name(dt), index);
  }

  void visit(ExternalPtrStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    const auto linear_index_name = stmt->short_name();
    const auto *argload = stmt->base_ptrs[0]->as<ArgLoadStmt>();
    const int arg_id = argload->arg_id;
    const int num_indices = stmt->indices.size();
    const auto &element_shape = stmt->element_shape;
    std::vector<std::string> size_var_names;

    const auto layout = stmt->element_dim <= 0 ? ExternalArrayLayout::kAOS
                                               : ExternalArrayLayout::kSOA;
    const size_t element_shape_index_offset =
        layout == ExternalArrayLayout::kAOS ? num_indices - element_shape.size()
                                            : 0;
    for (int i = 0; i < num_indices - element_shape.size(); i++) {
      used.buf_args = true;
      used.int32 = true;
      std::string var_name = fmt::format("_s{}_{}{}", i, "arr", arg_id);

      if (!loaded_args_.count(var_name)) {
        emit("int {} = _args_i32_[{} + {} * {} + {}];", var_name,
             taichi_opengl_extra_args_base / sizeof(int), arg_id,
             taichi_max_num_indices, i);
        loaded_args_.insert(var_name);
      }
      size_var_names.push_back(std::move(var_name));
    }

    emit("int {} = {};", linear_index_name,
         num_indices == 0 ? "0" : stmt->indices[0]->short_name());

    size_t size_var_name_index = (layout == ExternalArrayLayout::kAOS) ? 1 : 0;
    for (int i = 1; i < num_indices; i++) {
      if (i >= element_shape_index_offset &&
          i < element_shape_index_offset + element_shape.size()) {
        emit("{} *= {};", linear_index_name,
             std::to_string(element_shape[i - element_shape_index_offset]));
      } else {
        emit("{} *= {};", linear_index_name,
             size_var_names[size_var_name_index++]);
      }
      emit("{} += {};", linear_index_name, stmt->indices[i]->short_name());
    }

    ptr_signats_[stmt->id] = "arr" + std::to_string(arg_id);
  }

  void visit(DecorationStmt *stmt) override {
  }

  void visit(UnaryOpStmt *stmt) override {
    auto dt_name = opengl_data_type_name(stmt->element_type());
    if (stmt->op_type == UnaryOpType::logic_not) {
      emit("{} {} = {}({} == 0);", dt_name, stmt->short_name(), dt_name,
           stmt->operand->short_name());
    } else if (stmt->op_type == UnaryOpType::neg) {
      emit("{} {} = {}(-{});", dt_name, stmt->short_name(), dt_name,
           stmt->operand->short_name());
    } else if (stmt->op_type == UnaryOpType::rsqrt) {
      emit("{} {} = {}(inversesqrt({}));", dt_name, stmt->short_name(), dt_name,
           stmt->operand->short_name());
    } else if (stmt->op_type == UnaryOpType::sgn) {
      emit("{} {} = {}(sign({}));", dt_name, stmt->short_name(), dt_name,
           stmt->operand->short_name());
    } else if (stmt->op_type == UnaryOpType::bit_not) {
      emit("{} {} = {}(~{});", dt_name, stmt->short_name(), dt_name,
           stmt->operand->short_name());
    } else if (stmt->op_type == UnaryOpType::cast_value) {
      emit("{} {} = {}({});", dt_name, stmt->short_name(),
           opengl_data_type_name(stmt->cast_type), stmt->operand->short_name());
    } else if (stmt->op_type == UnaryOpType::cast_bits) {
      constexpr int FLOATING_POINT = 0;
      constexpr int SIGNED_INTEGER = 1;
      constexpr int UNSIGNED_INTEGER = 2;

      auto dst_type = stmt->cast_type;
      auto src_type = stmt->operand->element_type();
      auto dst_type_id = FLOATING_POINT;
      if (is_integral(dst_type))
        dst_type_id = is_unsigned(dst_type) ? UNSIGNED_INTEGER : SIGNED_INTEGER;
      auto src_type_id = FLOATING_POINT;
      if (is_integral(src_type))
        src_type_id = is_unsigned(src_type) ? UNSIGNED_INTEGER : SIGNED_INTEGER;

      TI_ASSERT_INFO(
          data_type_size(dst_type) == data_type_size(src_type),
          "bit_cast is only supported between data type with same size");

      if (dst_type_id != FLOATING_POINT && src_type_id != FLOATING_POINT) {
        emit("{} {} = {}({});", dt_name, stmt->short_name(), dt_name,
             stmt->operand->short_name());

      } else if (dst_type_id == FLOATING_POINT &&
                 src_type_id == SIGNED_INTEGER) {
        emit("{} {} = intBitsToFloat({});", dt_name, stmt->short_name(),
             stmt->operand->short_name());

      } else if (dst_type_id == SIGNED_INTEGER &&
                 src_type_id == FLOATING_POINT) {
        emit("{} {} = floatBitsToInt({});", dt_name, stmt->short_name(),
             stmt->operand->short_name());

      } else if (dst_type_id == FLOATING_POINT &&
                 src_type_id == UNSIGNED_INTEGER) {
        emit("{} {} = uintBitsToFloat({});", dt_name, stmt->short_name(),
             stmt->operand->short_name());

      } else if (dst_type_id == UNSIGNED_INTEGER &&
                 src_type_id == FLOATING_POINT) {
        emit("{} {} = floatBitsToUint({});", dt_name, stmt->short_name(),
             stmt->operand->short_name());

      } else {
        TI_ERROR("[glsl] unsupported bit cast from {} to {}",
                 data_type_name(src_type), data_type_name(dst_type));
      }
    } else {
      emit("{} {} = {}({}({}));", dt_name, stmt->short_name(), dt_name,
           unary_op_type_name(stmt->op_type), stmt->operand->short_name());
    }
  }

  void visit(BinaryOpStmt *bin) override {
    const auto dt_name = opengl_data_type_name(bin->element_type());
    const auto lhs_name = bin->lhs->short_name();
    const auto rhs_name = bin->rhs->short_name();
    const auto bin_name = bin->short_name();
    if (bin->op_type == BinaryOpType::floordiv) {
      TI_WARN("floordiv called! It should be taken care by demote_operations");
      if (is_integral(bin->lhs->element_type()) &&
          is_integral(bin->rhs->element_type())) {
        emit(
            "{} {} = {}(sign({}) * {} >= 0 ? abs({}) / abs({}) : sign({}) * "
            "(abs({}) + abs({}) - 1) / {});",
            dt_name, bin_name, dt_name, lhs_name, rhs_name, lhs_name, rhs_name,
            lhs_name, lhs_name, rhs_name, rhs_name);
        return;
      }
      // NOTE: the 1e-6 here is for precision reason, or `7 // 7` will obtain 0
      // instead of 1
      emit(
          "{} {} = {}(floor((float({}) * (1 + sign({} * {}) * 1e-6)) / "
          "float({})));",
          dt_name, bin_name, dt_name, lhs_name, lhs_name, rhs_name, rhs_name);
      return;
    } else if (bin->op_type == BinaryOpType::mod) {
      // NOTE: the GLSL built-in function `mod()` is a pythonic mod: x - y *
      // floor(x / y)
      emit("{} {} = {} - {} * int({} / {});", dt_name, bin_name, lhs_name,
           rhs_name, lhs_name, rhs_name);
      // FIXME: hack! doesn't make too much difference on mobile.
      // emit("{} {} = {} & int({} - 1); // mod", dt_name, bin_name, lhs_name,
      // rhs_name);
      return;
    } else if (bin->op_type == BinaryOpType::atan2) {
      if (bin->element_type() ==
          PrimitiveType::f64) {  // don't know why no atan(double, double)
        emit("{} {} = {}(atan(float({}), float({})));", dt_name, bin_name,
             dt_name, lhs_name, rhs_name);
      } else {
        emit("{} {} = atan({}, {});", dt_name, bin_name, lhs_name, rhs_name);
      }
      return;
    } else if (bin->op_type == BinaryOpType::pow &&
               is_integral(bin->rhs->element_type())) {
      // The GLSL `pow` is not so percise for `int`... e.g.: `pow(5, 3)` obtains
      // 124 So that we have to use some hack to make it percise. Discussion:
      // https://github.com/taichi-dev/taichi/pull/943#issuecomment-626354902
      emit("{} {} = {}(fast_pow_{}({}, {}));", dt_name, bin_name, dt_name,
           opengl_data_type_short_name(bin->lhs->element_type()), lhs_name,
           rhs_name);
      used.fast_pow = true;
      return;
    }
    const auto binop = binary_op_type_symbol(bin->op_type);
    if (is_opengl_binary_op_infix(bin->op_type)) {
      if (is_opengl_binary_op_different_return_type(bin->op_type) ||
          bin->element_type() != bin->lhs->element_type() ||
          bin->element_type() != bin->rhs->element_type()) {
        if (is_comparison(bin->op_type)) {
          // TODO(#577): Taichi uses -1 as true due to LLVM i1... See
          // https://github.com/taichi-dev/taichi/blob/6989c0e21d437a9ffdc0151cee9d3aa2aaa2241d/taichi/codegen/codegen_llvm.cpp#L564
          // This is a workaround to make OpenGL compatible with the behavior.
          emit("{} {} = -{}({} {} {});", dt_name, bin_name, dt_name, lhs_name,
               binop, rhs_name);
        } else {
          emit("{} {} = {}({} {} {});", dt_name, bin_name, dt_name, lhs_name,
               binop, rhs_name);
        }
      } else {
        emit("{} {} = {} {} {};", dt_name, bin_name, lhs_name, binop, rhs_name);
      }
    } else {
      // This is a function call
      emit("{} {} = {}({}({}, {}));", dt_name, bin_name, dt_name, binop,
           lhs_name, rhs_name);
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt = stmt->dest->element_type().ptr_removed();

    auto val_name = stmt->val->short_name();

    emit("{} {};", opengl_data_type_name(stmt->val->element_type()),
         stmt->short_name());

    if (stmt->is_reduction &&
        (dt->is_primitive(PrimitiveTypeID::f32) ||
         dt->is_primitive(PrimitiveTypeID::i32) ||
         dt->is_primitive(PrimitiveTypeID::u32)) &&
        (stmt->op_type == AtomicOpType::add ||
         stmt->op_type == AtomicOpType::sub ||
         stmt->op_type == AtomicOpType::min ||
         stmt->op_type == AtomicOpType::max)) {
      used.reduction = true;
      val_name = stmt->short_name() + "_reduction";

      auto op_name = "";

      if (stmt->op_type == AtomicOpType::sub ||
          stmt->op_type == AtomicOpType::add) {
        op_name = "add";
      } else if (stmt->op_type == AtomicOpType::max) {
        op_name = "max";
      } else if (stmt->op_type == AtomicOpType::min) {
        op_name = "min";
      }

      emit("{} {} = reduction_workgroup_{}_{}({});",
           opengl_data_type_name(stmt->val->element_type()), val_name, op_name,
           opengl_data_type_name(stmt->val->element_type()),
           stmt->val->short_name());

      emit("if (gl_LocalInvocationIndex == 0)");
    }

    emit("{{ // Begin Atomic Op");

    if (maybe_generate_fatomics_using_nv_ext(stmt, dt, val_name)) {
      // Do nothing
    } else {
      if (dt != PrimitiveType::f32) {
        TI_ERROR(
            "unsupported atomic operation for PrimitiveType::{}, "
            "this may because your OpenGL is missing that extension, "
            "see `glewinfo` for more details",
            opengl_data_type_short_name(dt));
      }
      used.simulated_atomic_float = true;
      used.int32 = true;  // since simulated atomics are based on _data_i32_
      std::string index =
          stmt->dest->is<ExternalPtrStmt>()
              ? stmt->dest->short_name()
              : fmt::format("{} >> {}", stmt->dest->short_name(),
                            opengl_data_address_shifter(dt));
      emit("{} = {}_{}_{}({}, {});", stmt->short_name(),
           opengl_atomic_op_type_cap_name(stmt->op_type),
           ptr_signats_.at(stmt->dest->id), opengl_data_type_short_name(dt),
           index, val_name);
    }

    emit("}} // End Atomic Op");
  }

  bool maybe_generate_fatomics_using_nv_ext(AtomicOpStmt *stmt,
                                            DataType dt,
                                            const std::string &val_name) {
    if (!allows_nv_shader_ext_ && !dt->is_primitive(PrimitiveTypeID::i32)) {
      return false;
    }
    const bool check_int =
        (dt->is_primitive(PrimitiveTypeID::i32) ||
         (TI_OPENGL_REQUIRE(used, GL_NV_shader_atomic_int64) &&
          dt->is_primitive(PrimitiveTypeID::i64)));
    const bool check_add = (stmt->op_type == AtomicOpType::add ||
                            stmt->op_type == AtomicOpType::sub);
    const bool check_float =
        ((TI_OPENGL_REQUIRE(used, GL_NV_shader_atomic_float) &&
          dt->is_primitive(PrimitiveTypeID::f32)) ||
         (TI_OPENGL_REQUIRE(used, GL_NV_shader_atomic_float64) &&
          dt->is_primitive(PrimitiveTypeID::f64)));
    if (check_int || (check_add && check_float)) {
      std::string index =
          stmt->dest->is<ExternalPtrStmt>()
              ? stmt->dest->short_name()
              : fmt::format("{} >> {}", stmt->dest->short_name(),
                            opengl_data_address_shifter(dt));

      emit("{} = {}(_{}_{}_[{}], {});", stmt->short_name(),
           opengl_atomic_op_type_cap_name(stmt->op_type),
           ptr_signats_.at(stmt->dest->id), opengl_data_type_short_name(dt),
           index, val_name);
      return true;
    }
    return false;
  }

  void visit(TernaryOpStmt *tri) override {
    TI_ASSERT(tri->op_type == TernaryOpType::select);
    emit("{} {} = {} != 0 ? {} : {};",
         opengl_data_type_name(tri->element_type()), tri->short_name(),
         tri->op1->short_name(), tri->op2->short_name(),
         tri->op3->short_name());
  }

  void visit(LocalLoadStmt *stmt) override {
    bool linear_index = true;
    for (int i = 0; i < (int)stmt->src.size(); i++) {
      if (stmt->src[i].offset != i) {
        linear_index = false;
      }
    }
    if (stmt->same_source() && linear_index &&
        stmt->width() == stmt->src[0].var->width()) {
      auto src = stmt->src[0].var;
      emit("{} {} = {};", opengl_data_type_name(stmt->element_type()),
           stmt->short_name(), src->short_name());
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    emit("{} = {};", stmt->dest->short_name(), stmt->val->short_name());
  }

  void visit(AllocaStmt *alloca) override {
    auto dt_name = opengl_data_type_name(alloca->element_type());
    emit("{} {} = {}(0);", dt_name, alloca->short_name(), dt_name);
  }

  void visit(ConstStmt *const_stmt) override {
    TI_ASSERT(const_stmt->width() == 1);
    auto dt_name = opengl_data_type_name(const_stmt->element_type());
    emit("{} {} = {}({});", dt_name, const_stmt->short_name(), dt_name,
         const_stmt->val[0].stringify());
  }

  void visit(ReturnStmt *stmt) override {
    used.buf_args = true;
    // TODO: use stmt->ret_id instead of 0 as index
    int idx{0};
    for (auto &value : stmt->values) {
      emit("_args_{}_[({} >> {}) + {}] = {};",
           opengl_data_type_short_name(value->element_type()),
           taichi_opengl_ret_base,
           opengl_data_address_shifter(value->element_type()), idx,
           value->short_name());
      idx += (4 - opengl_data_address_shifter(value->element_type()));
      // opengl only support i32, f32 and f64 array, but there are 64bit slots
      // in taichi's result buffer,so we need two slots to make them match.
    }
  }

  void visit(ArgLoadStmt *stmt) override {
    const auto dt = opengl_data_type_name(stmt->element_type());
    if (stmt->is_ptr) {
      if (!used.arr_arg_to_bind_idx.count(stmt->arg_id)) {
        used.arr_arg_to_bind_idx[stmt->arg_id] =
            static_cast<int>(GLBufId::Arr) + stmt->arg_id;
      }
    } else {
      used.buf_args = true;
      emit("{} {} = _args_{}_[{} << {}];", dt, stmt->short_name(),
           opengl_data_type_short_name(stmt->element_type()), stmt->arg_id,
           opengl_argument_address_shifter(stmt->element_type()));
    }
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    TI_ASSERT(stmt->type == ExternalFuncCallStmt::ASSEMBLY);
    auto format = stmt->asm_source;
    std::string source;

    for (int i = 0; i < format.size(); i++) {
      char c = format[i];
      if (c == '%' || c == '$') {  // '$' for output, '%' for input
        int num = 0;
        while (i < format.size()) {
          i += 1;
          if (!::isdigit(format[i])) {
            i -= 1;
            break;
          }
          num *= 10;
          num += format[i] - '0';
        }
        auto args = (c == '%') ? stmt->arg_stmts : stmt->output_stmts;
        TI_ASSERT_INFO(num < args.size(), "{}{} out of {} argument range {}", c,
                       num, ((c == '%') ? "input" : "output"), args.size());
        source += args[num]->short_name();
      } else {
        source.push_back(c);
      }
    }

    emit("{};", source);
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    const auto name = stmt->short_name();
    const auto arg_id = stmt->arg_id;
    const auto axis = stmt->axis;
    used.buf_args = true;
    used.int32 = true;
    if (!loaded_args_.count(name)) {
      emit("int {} = _args_i32_[{} + {} * {} + {}];", name,
           taichi_opengl_extra_args_base / sizeof(int), arg_id,
           taichi_max_num_indices, axis);
      loaded_args_.insert(name);
    }
  }

  std::string make_kernel_name() {
    return fmt::format("{}{}", glsl_kernel_prefix_, glsl_kernel_count_++);
  }

  void generate_serial_kernel(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::serial);
    const std::string glsl_kernel_name = make_kernel_name();
    this->glsl_kernel_name_ = glsl_kernel_name;
    emit("void {}()", glsl_kernel_name);
    emit("{{ // serial");
    stmt->body->accept(this);
    emit("}}\n");
  }

  void generate_grid_stride_loop_header() {
    ScopedIndent _s(line_appender_);
  }

  struct ScopedGridStrideLoop {
    KernelGen *gen;
    std::unique_ptr<ScopedIndent> s;

    ScopedGridStrideLoop(KernelGen *gen, int const_iterations)
        : ScopedGridStrideLoop(gen,
                               fmt::format("{}", const_iterations),
                               const_iterations) {
    }

    ScopedGridStrideLoop(KernelGen *gen,
                         std::string iterations,
                         int const_iterations = -1)
        : gen(gen) {
      gen->emit("int _sid0 = int(gl_GlobalInvocationID.x);");
      gen->emit("for (int _sid = _sid0; _sid < ({}); _sid += {}) {{",
                iterations, "int(gl_WorkGroupSize.x * gl_NumWorkGroups.x)");
      s = std::make_unique<ScopedIndent>(gen->line_appender_);

      if (gen->num_workgroups_ == 0) {
        // if not specified, guess an optimal grid_dim for different situations
        // Refs:
        // https://stackoverflow.com/questions/36374652/compute-shaders-optimal-data-division-on-invocations-threads-and-workgroups
        if (const_iterations > 0) {
          if (gen->used_tls_) {
            // const range with TLS reduction
            gen->num_workgroups_ = std::max(
                const_iterations / std::max(gen->workgroup_size_, 1) / 32, 1);
            gen->workgroup_size_ = std::max(gen->workgroup_size_ / 4, 1);
          } else {
            // const range
            gen->num_workgroups_ =
                std::max((const_iterations + gen->workgroup_size_ - 1) /
                             gen->workgroup_size_,
                         1);
          }
        } else {
          // dynamic range
          // TODO(archibate): think for a better value for SM utilization:
          gen->num_workgroups_ = 256;
        }
      }
    }

    ~ScopedGridStrideLoop() {
      s = nullptr;
      gen->emit("}}");
    }
  };
  void gen_array_range(Stmt *stmt) {
    int num_operands = stmt->num_operands();
    for (int i = 0; i < num_operands; i++) {
      gen_array_range(stmt->operand(i));
    }
    stmt->accept(this);
  }
  void generate_range_for_kernel(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::range_for);
    const std::string glsl_kernel_name = make_kernel_name();
    emit("void {}()", glsl_kernel_name);
    this->glsl_kernel_name_ = glsl_kernel_name;
    emit("{{ // range for");

    used_tls_ = (stmt->tls_prologue != nullptr);
    if (used_tls_) {
      auto tls_size = stmt->tls_size;
      // TODO(k-ye): support 'cursor' in LineAppender:
      emit("int _tls_i32_[{}];", (tls_size + 3) / 4);
      if (used.int64)
        emit("int64_t _tls_i64_[{}];", (tls_size + 7) / 8);
      if (used.uint32)
        emit("int _tls_u32_[{}];", (tls_size + 3) / 4);
      if (used.uint64)
        emit("int64_t _tls_u64_[{}];", (tls_size + 7) / 8);
      emit("float _tls_f32_[{}];", (tls_size + 3) / 4);
      if (used.float64)
        emit("double _tls_f64_[{}];", (tls_size + 7) / 8);
      emit("{{  // TLS prologue");
      stmt->tls_prologue->accept(this);
      emit("}}");
    }

    if (stmt->const_begin && stmt->const_end) {
      ScopedIndent _s(line_appender_);
      emit("// range known at compile time");
      auto begin_value = stmt->begin_value;
      auto end_value = stmt->end_value;
      if (end_value < begin_value)
        end_value = begin_value;
      workgroup_size_ = stmt->block_dim;
      num_workgroups_ = stmt->grid_dim;
      ScopedGridStrideLoop _gsl(this, end_value - begin_value);
      emit("int _itv = {} + _sid;", begin_value);
      // range_hint is known after compilation, e.g. range of field
      stmt->range_hint = std::to_string(end_value - begin_value);
      stmt->body->accept(this);
    } else {
      ScopedIndent _s(line_appender_);
      std::string begin_expr, end_expr;
      if (stmt->end_stmt) {
        emit("// range from args buffer");
        TI_ASSERT(stmt->const_begin);
        begin_expr = std::to_string(stmt->begin_value);
        gen_array_range(stmt->end_stmt);
        end_expr = stmt->end_stmt->short_name();
      } else {
        emit("// range known at runtime");
        begin_expr = stmt->const_begin ? std::to_string(stmt->begin_value)
                                       : fmt::format("_gtmp_i32_[{} >> 2]",
                                                     stmt->begin_offset);
        end_expr = stmt->const_end
                       ? std::to_string(stmt->end_value)
                       : fmt::format("_gtmp_i32_[{} >> 2]", stmt->end_offset);
      }
      workgroup_size_ = stmt->block_dim;
      num_workgroups_ = stmt->grid_dim;
      emit("int _beg = {}, _end = {};", begin_expr, end_expr);
      ScopedGridStrideLoop _gsl(this, "_end - _beg");
      emit("int _itv = _beg + _sid;");
      stmt->body->accept(this);
    }

    if (used_tls_) {
      TI_ASSERT(stmt->tls_epilogue != nullptr);
      emit("{{  // TLS epilogue");
      stmt->tls_epilogue->accept(this);
      emit("}}");
    }
    used_tls_ = false;

    emit("}}\n");
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    used.buf_gtmp = true;
    emit("int {} = {};", stmt->short_name(), stmt->offset);
    ptr_signats_[stmt->id] = "gtmp";
  }

  void visit(ThreadLocalPtrStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("int {} = {};", stmt->short_name(), stmt->offset);
    ptr_signats_[stmt->id] = "tls";
  }

  void visit(LoopIndexStmt *stmt) override {
    TI_ASSERT(stmt->index == 0);  // TODO: multiple indices
    if (stmt->loop->is<OffloadedStmt>()) {
      auto type = stmt->loop->as<OffloadedStmt>()->task_type;
      if (type == OffloadedStmt::TaskType::range_for) {
        emit("int {} = _itv;", stmt->short_name());
      } else if (type == OffloadedStmt::TaskType::struct_for) {
        emit("int {} = _itv; // struct for", stmt->short_name());
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->loop->is<RangeForStmt>()) {
      emit("int {} = {};", stmt->short_name(), stmt->loop->short_name());
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(RangeForStmt *for_stmt) override {
    TI_ASSERT(for_stmt->width() == 1);
    auto loop_var_name = for_stmt->short_name();
    if (!for_stmt->reversed) {
      emit("for (int {}_ = {}; {}_ < {}; {}_ += {}) {{", loop_var_name,
           for_stmt->begin->short_name(), loop_var_name,
           for_stmt->end->short_name(), loop_var_name, 1);
      emit("  int {} = {}_;", loop_var_name, loop_var_name);
    } else {
      // reversed for loop
      emit("for (int {}_ = {} - {}; {}_ >= {}; {}_ -= {}) {{", loop_var_name,
           for_stmt->end->short_name(), 1, loop_var_name,
           for_stmt->begin->short_name(), loop_var_name, 1);
      emit("  int {} = {}_;", loop_var_name, loop_var_name);
    }
    for_stmt->body->accept(this);
    emit("}}");
  }

  void visit(WhileControlStmt *stmt) override {
    emit("if ({} == 0) break;", stmt->cond->short_name());
  }

  void visit(ContinueStmt *stmt) override {
    // stmt->as_return() is unused when embraced with a grid-stride-loop
    emit("continue;");
  }

  void visit(WhileStmt *stmt) override {
    emit("while (true) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(OffloadedStmt *stmt) override {
    auto map = irpass::detect_external_ptr_access_in_task(stmt);

    this->extptr_access_ = std::move(map);

    generate_header();
    TI_ASSERT(is_top_level_);
    is_top_level_ = false;
    const auto task_type = stmt->task_type;
    if (task_type == OffloadedTaskType::serial) {
      generate_serial_kernel(stmt);
    } else if (task_type == OffloadedTaskType::range_for) {
      generate_range_for_kernel(stmt);
    } else {
      // struct_for is automatically lowered to ranged_for for dense snodes
      // (#378). So we only need to support serial and range_for tasks.
      TI_ERROR("[glsl] Unsupported offload type={} on OpenGL arch",
               stmt->task_name());
    }
    is_top_level_ = true;
    generate_task_bottom(task_type, stmt->range_hint);
    loaded_args_.clear();
  }

  void visit(StructForStmt *) override {
    TI_ERROR("[glsl] Struct for cannot be nested under OpenGL for now");
  }

  void visit(IfStmt *if_stmt) override {
    emit("if ({} != 0) {{", if_stmt->cond->short_name());
    if (if_stmt->true_statements) {
      if_stmt->true_statements->accept(this);
    }
    if (if_stmt->false_statements) {
      emit("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    emit("}}");
  }

 public:
  CompiledTaichiKernel get_compiled_program() {
    // We have to set it at the last moment, to get all used feature.
    compiled_program_.set_used(used);
    return std::move(compiled_program_);
  }

  void run() {
    kernel_->ir->accept(this);
  }
};

}  // namespace

CompiledTaichiKernel OpenglCodeGen::gen(void) {
#if defined(TI_WITH_OPENGL)
  KernelGen codegen(kernel_, struct_compiled_, kernel_name_,
                    allows_nv_shader_ext_);
  codegen.run();
  return codegen.get_compiled_program();
#else
  TI_NOT_IMPLEMENTED
#endif
}

void OpenglCodeGen::lower() {
  auto ir = kernel_->ir.get();
  auto &config = kernel_->program->config;
  config.demote_dense_struct_fors = true;
  irpass::compile_to_executable(ir, config, kernel_,
                                /*autodiff_mode=*/kernel_->autodiff_mode,
                                /*ad_use_stack=*/false, config.print_ir,
                                /*lower_global_access=*/true,
                                /*make_thread_local=*/config.make_thread_local);
#ifdef _GLSL_DEBUG
  irpass::print(ir);
#endif
}

CompiledTaichiKernel OpenglCodeGen::compile(Kernel &kernel) {
  this->kernel_ = &kernel;

  this->lower();
  return this->gen();
}

}  // namespace opengl
TLANG_NAMESPACE_END
