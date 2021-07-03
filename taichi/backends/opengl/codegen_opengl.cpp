//#define _GLSL_DEBUG 1
#include "codegen_opengl.h"

#include <string>

#include "taichi/backends/opengl/opengl_api.h"
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
#define TI_INSIDE_OPENGL_CODEGEN
#include "taichi/backends/opengl/shaders/atomics_macro_f32.glsl.h"
#include "taichi/backends/opengl/shaders/runtime.h"
#include "taichi/backends/opengl/shaders/listman.h"
#include "taichi/backends/opengl/shaders/random.glsl.h"
#include "taichi/backends/opengl/shaders/fast_pow.glsl.h"
#include "taichi/backends/opengl/shaders/print.glsl.h"
#undef TI_INSIDE_OPENGL_CODEGEN
}  // namespace shaders

int find_children_id(const SNode *snode) {
  auto parent = snode->parent;
  for (int i = 0; i < parent->ch.size(); i++) {
    if (parent->ch[i].get() == snode)
      return i;
  }
  TI_ERROR("Child not found in parent!");
}

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

class KernelGen : public IRVisitor {
  Kernel *kernel;

 public:
  KernelGen(Kernel *kernel,
            std::string kernel_name,
            StructCompiledResult *struct_compiled)
      : kernel(kernel),
        struct_compiled_(struct_compiled),
        kernel_name_(kernel_name),
        glsl_kernel_prefix_(kernel_name),
        compiled_program_(std::make_unique<CompiledProgram>(kernel)),
        ps(std::make_unique<ParallelSize>()) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

 private:
  // constants:
  StructCompiledResult *struct_compiled_;
  const SNode *root_snode_;
  GetRootStmt *root_stmt_;
  std::string kernel_name_;
  std::string root_snode_type_name_;
  std::string glsl_kernel_prefix_;

  // throughout variables:
  int glsl_kernel_count_{0};
  bool is_top_level_{true};
  std::unique_ptr<CompiledProgram> compiled_program_;
  UsedFeature used;  // TODO: is this actually per-offload?

  // per-offload variables:
  LineAppender line_appender_;
  LineAppender line_appender_header_;
  std::string glsl_kernel_name_;
  std::unique_ptr<ParallelSize> ps;
  bool used_tls;  // TODO: move into UsedFeature?

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    line_appender_.append(std::move(f), std::move(args)...);
  }

  void generate_header() {
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

  void generate_bottom() {
    // TODO(archibate): <kernel_name>() really necessary? How about just main()?
    emit("void main()");
    emit("{{");
    if (used.random)
      emit("  _init_rand();");
    if (glsl_kernel_name_.size())
      emit("  {}();", glsl_kernel_name_);
    emit("}}");

    // clang-format off
    if (used.print)  // the runtime buffer is only used for print now..
      line_appender_header_.append_raw(shaders::kOpenGLRuntimeSourceCode);
    if (used.listman)
      line_appender_header_.append_raw(shaders::kOpenGLListmanSourceCode);

    std::string kernel_header;
#define DEFINE_LAYOUT(name, id, dt, dtype) \
      kernel_header += "layout(std430, binding = " + fmt::format("{}", id) \
                    + ") buffer " #name "_" #dt " { " #dtype " _" \
                    #name "_" #dt "_[]; };\n"
#define REGISTER_BUFFER(name, id) do { \
    if (used.int32) DEFINE_LAYOUT(name, id, i32, int); \
    if (used.int64) DEFINE_LAYOUT(name, id, i64, int64_t); \
    if (used.uint32) DEFINE_LAYOUT(name, id, u32, uint); \
    if (used.uint64) DEFINE_LAYOUT(name, id, u64, uint64_t); \
    if (used.float32) DEFINE_LAYOUT(name, id, f32, float); \
    if (used.float64) DEFINE_LAYOUT(name, id, f64, double); \
  } while (0)

    REGISTER_BUFFER(data, GLBufId::Root);
    if (used.buf_gtmp)
      REGISTER_BUFFER(gtmp, GLBufId::Gtmp);
    if (used.buf_args)
      REGISTER_BUFFER(args, GLBufId::Args);
    if (used.buf_extr)
      REGISTER_BUFFER(extr, GLBufId::Extr);

#undef REGISTER_BUFFER
#undef DEFINE_LAYOUT
    // clang-format on

    if (used.simulated_atomic_float) {
      line_appender_header_.append_raw(shaders::kOpenGLAtomicF32SourceCode);
      kernel_header += ("DEFINE_ATOMIC_F32_FUNCTIONS(data);\n");
      if (used.buf_gtmp) {
        kernel_header += ("DEFINE_ATOMIC_F32_FUNCTIONS(gtmp);\n");
      }
      if (used.buf_extr) {
        kernel_header += ("DEFINE_ATOMIC_F32_FUNCTIONS(extr);\n");
      }
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
        "#version 430 core\n" + extensions + "precision highp float;\n" +
        line_appender_header_.lines() + line_appender_.lines();
    compiled_program_->add(std::move(glsl_kernel_name_), kernel_src_code,
                           std::move(ps));
    auto &config = kernel->program->config;
    if (config.print_kernel_llvm_ir) {
      static FileSequenceWriter writer("shader{:04d}.comp",
                                       "OpenGL compute shader");
      writer.write(kernel_src_code);
    }
    line_appender_header_.clear_all();
    line_appender_.clear_all();
    ps = std::make_unique<ParallelSize>();
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

  virtual void visit(Stmt *stmt) override {
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
        int stridx = compiled_program_->lookup_or_add_string(str);
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
      if (stmt->snode->type == SNodeType::dense) {
        // do nothing
      } else if (stmt->snode->type == SNodeType::dynamic) {
        used.int32 = true;
        emit("atomicMax(_data_i32_[{} >> 2], {} + 1); // dynamic activate",
             get_snode_meta_address(stmt->snode),
             stmt->input_index->short_name());
      } else {
        TI_NOT_IMPLEMENTED
      }
    }
  }

  void visit(SNodeOpStmt *stmt) override {  // IAPR?
    if (stmt->op_type == SNodeOpType::activate) {
      if (stmt->snode->type == SNodeType::dense ||
          stmt->snode->type == SNodeType::root) {
        // do nothing
      } else if (stmt->snode->type == SNodeType::dynamic) {
        used.int32 = true;
        emit("atomicMax(_data_i32_[{} >> 2], {} + 1); // dynamic activate",
             get_snode_meta_address(stmt->snode), stmt->val->short_name());
      } else {
        TI_NOT_IMPLEMENTED
      }

    } else if (stmt->op_type == SNodeOpType::deactivate) {
      if (stmt->snode->type == SNodeType::dense ||
          stmt->snode->type == SNodeType::root) {
        // do nothing
      } else if (stmt->snode->type == SNodeType::dynamic) {
        used.int32 = true;
        emit("_data_i32_[{} >> 2] = 0; // dynamic deactivate",
             get_snode_meta_address(stmt->snode), stmt->val->short_name());
      } else {
        TI_NOT_IMPLEMENTED
      }

    } else if (stmt->op_type == SNodeOpType::is_active) {
      TI_ASSERT(stmt->ret_type->is_primitive(PrimitiveTypeID::i32));
      if (stmt->snode->type == SNodeType::dense ||
          stmt->snode->type == SNodeType::root) {
        emit("int {} = 1;", stmt->short_name());
      } else if (stmt->snode->type == SNodeType::dynamic) {
        used.int32 = true;
        emit("int {} = int({} < _data_i32_[{} >> 2]);", stmt->short_name(),
             stmt->val->short_name(), get_snode_meta_address(stmt->snode));
      } else {
        TI_NOT_IMPLEMENTED
      }

    } else if (stmt->op_type == SNodeOpType::append) {
      TI_ASSERT(stmt->snode->type == SNodeType::dynamic);
      TI_ASSERT(stmt->ret_type->is_primitive(PrimitiveTypeID::i32));
      used.int32 = true;
      emit("int {} = atomicAdd(_data_i32_[{} >> 2], 1);", stmt->short_name(),
           get_snode_meta_address(stmt->snode));
      auto dt = stmt->val->element_type();
      emit("int _ad_{} = {} + {} * {};", stmt->short_name(),
           get_snode_base_address(stmt->snode), stmt->short_name(),
           struct_compiled_->snode_map.at(stmt->snode->node_type_name)
               .elem_stride);
      emit("_data_{}_[_ad_{} >> {}] = {};", opengl_data_type_short_name(dt),
           stmt->short_name(), opengl_data_address_shifter(dt),
           stmt->val->short_name());

    } else if (stmt->op_type == SNodeOpType::length) {
      TI_ASSERT(stmt->snode->type == SNodeType::dynamic);
      TI_ASSERT(stmt->ret_type->is_primitive(PrimitiveTypeID::i32));
      used.int32 = true;
      emit("int {} = _data_i32_[{} >> 2];", stmt->short_name(),
           get_snode_meta_address(stmt->snode));

    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  std::map<int, std::string> ptr_signats;

  void visit(GetChStmt *stmt) override {
    emit("int {} = {} + {}; // {}", stmt->short_name(),
         stmt->input_ptr->short_name(),
         struct_compiled_->snode_map.at(stmt->input_snode->node_type_name)
             .children_offsets[stmt->chid],
         stmt->output_snode->node_type_name);
    if (stmt->output_snode->is_place())
      ptr_signats[stmt->id] = "data";
  }

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt = stmt->val->element_type();
    emit("_{}_{}_[{} >> {}] = {};",
         ptr_signats.at(stmt->dest->id),  // throw out_of_range if not a pointer
         opengl_data_type_short_name(dt), stmt->dest->short_name(),
         opengl_data_address_shifter(dt), stmt->val->short_name());
  }

  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt = stmt->element_type();
    emit("{} {} = _{}_{}_[{} >> {}];",
         opengl_data_type_name(stmt->element_type()), stmt->short_name(),
         ptr_signats.at(stmt->src->id), opengl_data_type_short_name(dt),
         stmt->src->short_name(), opengl_data_address_shifter(dt));
  }

  void visit(ExternalPtrStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    const auto linear_index_name = fmt::format("_li_{}", stmt->short_name());
    emit("int {} = 0;", linear_index_name);
    emit("{{ // linear seek");
    {
      ScopedIndent _s(line_appender_);
      const auto *argload = stmt->base_ptrs[0]->as<ArgLoadStmt>();
      const int arg_id = argload->arg_id;
      const int num_indices = stmt->indices.size();
      std::vector<std::string> size_var_names;
      for (int i = 0; i < num_indices; i++) {
        used.buf_args = true;
        used.int32 = true;
        std::string var_name = fmt::format("_s{}_{}", i, stmt->short_name());
        emit("int {} = _args_i32_[{} + {} * {} + {}];", var_name,
             taichi_opengl_earg_base / sizeof(int), arg_id,
             taichi_max_num_indices, i);
        size_var_names.push_back(std::move(var_name));
      }
      for (int i = 0; i < num_indices; i++) {
        emit("{} *= {};", linear_index_name, size_var_names[i]);
        emit("{} += {};", linear_index_name, stmt->indices[i]->short_name());
      }
    }
    emit("}}");

    emit("int {} = {} + ({} << {});", stmt->short_name(),
         stmt->base_ptrs[0]->short_name(), linear_index_name,
         opengl_data_address_shifter(stmt->base_ptrs[0]->element_type()));
    used.buf_extr = true;
    ptr_signats[stmt->id] = "extr";
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
    if (dt->is_primitive(PrimitiveTypeID::i32) ||
        (TI_OPENGL_REQUIRE(used, GL_NV_shader_atomic_int64) &&
         dt->is_primitive(PrimitiveTypeID::i64)) ||
        ((stmt->op_type == AtomicOpType::add ||
          stmt->op_type == AtomicOpType::sub) &&
         ((TI_OPENGL_REQUIRE(used, GL_NV_shader_atomic_float) &&
           dt->is_primitive(PrimitiveTypeID::f32)) ||
          (TI_OPENGL_REQUIRE(used, GL_NV_shader_atomic_float64) &&
           dt->is_primitive(PrimitiveTypeID::f64))))) {
      emit("{} {} = {}(_{}_{}_[{} >> {}], {});",
           opengl_data_type_name(stmt->val->element_type()), stmt->short_name(),
           opengl_atomic_op_type_cap_name(stmt->op_type),
           ptr_signats.at(stmt->dest->id), opengl_data_type_short_name(dt),
           stmt->dest->short_name(), opengl_data_address_shifter(dt),
           stmt->val->short_name());
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
      emit("{} {} = {}_{}_{}({} >> {}, {});",
           opengl_data_type_name(stmt->val->element_type()), stmt->short_name(),
           opengl_atomic_op_type_cap_name(stmt->op_type),
           ptr_signats.at(stmt->dest->id), opengl_data_type_short_name(dt),
           stmt->dest->short_name(), opengl_data_address_shifter(dt),
           stmt->val->short_name());
    }
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
    // TODO: consider use _rets_{}_ instead of _args_{}_
    // TODO: use stmt->ret_id instead of 0 as index
    emit("_args_{}_[0] = {};",
         opengl_data_type_short_name(stmt->element_type()),
         stmt->value->short_name());
  }

  void visit(ArgLoadStmt *stmt) override {
    const auto dt = opengl_data_type_name(stmt->element_type());
    used.buf_args = true;
    if (stmt->is_ptr) {
      used.int32 = true;
      emit("int {} = _args_i32_[{} << 1]; // is ext pointer {}",
           stmt->short_name(), stmt->arg_id, dt);
    } else {
      emit("{} {} = _args_{}_[{} << {}];", dt, stmt->short_name(),
           opengl_data_type_short_name(stmt->element_type()), stmt->arg_id,
           opengl_argument_address_shifter(stmt->element_type()));
    }
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    TI_ASSERT(!stmt->func);
    auto format = stmt->source;
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
    emit("int {} = _args_i32_[{} + {} * {} + {}];", name,
         taichi_opengl_earg_base / sizeof(int), arg_id, taichi_max_num_indices,
         axis);
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

      TI_ASSERT(gen->ps);
      if (gen->ps->grid_dim == 0) {
        // if not specified, guess an optimal grid_dim for different situations
        // Refs:
        // https://stackoverflow.com/questions/36374652/compute-shaders-optimal-data-division-on-invocations-threads-and-workgroups
        if (const_iterations > 0) {
          if (gen->used_tls) {
            // const range with TLS reduction
            gen->ps->grid_dim =
                std::max((size_t)const_iterations /
                             std::max(gen->ps->block_dim, (size_t)1) / 32,
                         (size_t)1);
            gen->ps->block_dim = std::max(gen->ps->block_dim / 4, (size_t)1);
          } else {
            // const range
            gen->ps->grid_dim =
                std::max(((size_t)const_iterations + gen->ps->block_dim - 1) /
                             gen->ps->block_dim,
                         (size_t)1);
          }
        } else {
          // dynamic range
          // TODO(archibate): think for a better value for SM utilization:
          gen->ps->grid_dim = 256;
        }
      }
    }

    ~ScopedGridStrideLoop() {
      s = nullptr;
      gen->emit("}}");
    }
  };

  void generate_range_for_kernel(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::range_for);
    const std::string glsl_kernel_name = make_kernel_name();
    emit("void {}()", glsl_kernel_name);
    this->glsl_kernel_name_ = glsl_kernel_name;
    emit("{{ // range for");

    used_tls = (stmt->tls_prologue != nullptr);
    if (used_tls) {
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
      ps = std::make_unique<ParallelSize>(stmt->block_dim, stmt->grid_dim);
      ScopedGridStrideLoop _gsl(this, end_value - begin_value);
      emit("int _itv = {} + _sid;", begin_value);
      stmt->body->accept(this);
    } else {
      ScopedIndent _s(line_appender_);
      emit("// range known at runtime");
      auto begin_expr = stmt->const_begin ? std::to_string(stmt->begin_value)
                                          : fmt::format("_gtmp_i32_[{} >> 2]",
                                                        stmt->begin_offset);
      auto end_expr = stmt->const_end ? std::to_string(stmt->end_value)
                                      : fmt::format("_gtmp_i32_[{} >> 2]",
                                                    stmt->end_offset);
      ps = std::make_unique<ParallelSize>(stmt->block_dim, stmt->grid_dim);
      emit("int _beg = {}, _end = {};", begin_expr, end_expr);
      ScopedGridStrideLoop _gsl(this, "_end - _beg");
      emit("int _itv = _beg + _sid;");
      stmt->body->accept(this);
    }

    if (used_tls) {
      TI_ASSERT(stmt->tls_epilogue != nullptr);
      emit("{{  // TLS epilogue");
      stmt->tls_epilogue->accept(this);
      emit("}}");
    }
    used_tls = false;

    emit("}}\n");
  }

  void generate_struct_for_kernel(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::struct_for);
    used.listman = true;
    const std::string glsl_kernel_name = make_kernel_name();
    emit("void {}()", glsl_kernel_name);
    this->glsl_kernel_name_ = glsl_kernel_name;
    emit("{{ // struct for {}", stmt->snode->node_type_name);
    {
      ScopedIndent _s(line_appender_);
      ps = std::make_unique<ParallelSize>(stmt->block_dim, stmt->grid_dim);
      ScopedGridStrideLoop _gsl(this, "_list_len_");
      emit("int _itv = _list_[_sid];");
      stmt->body->accept(this);
    }
    emit("}}\n");
  }

  size_t get_snode_base_address(const SNode *snode) {
    if (snode->type == SNodeType::root)
      return 0;
    int chid = find_children_id(snode);
    const auto &parent_meta =
        struct_compiled_->snode_map.at(snode->parent->node_type_name);
    auto choff = parent_meta.children_offsets[chid];
    return choff + get_snode_base_address(snode->parent);
  }

  size_t get_snode_meta_address(const SNode *snode) {
    auto addr = get_snode_base_address(snode);
    addr += struct_compiled_->snode_map.at(snode->node_type_name).stride;
    addr -= opengl_get_snode_meta_size(*snode);
    return addr;
  }

  void generate_listgen_for_dynamic(const SNode *snode) {
    TI_ASSERT(snode->type == SNodeType::dynamic);
    // the `length` field of a dynamic SNode is at it's end:
    // | x[0] | x[1] | x[2] | x[3] | ... | len |
    TI_ASSERT_INFO(snode->parent->type == SNodeType::root,
                   "Non-top-level dynamic not supported yet on OpenGL");
    size_t addr = get_snode_meta_address(snode);
    used.int32 = true;
    emit("_list_len_ = _data_i32_[{} >> 2];", addr);
    emit("for (int i = 0; i < _list_len_; i++) {{");
    {
      ScopedIndent _s(line_appender_);
      emit("_list_[i] = i;");
    }
    emit("}}");
  }

  void generate_listgen_for_dense(const SNode *snode) {
    TI_ASSERT(snode->type == SNodeType::dense);
    // the `length` field of a dynamic SNode is at it's end:
    // | x[0] | x[1] | x[2] | x[3] | ... | len |
    emit("_list_len_ = {};",
         struct_compiled_->snode_map[snode->node_type_name].length);
    emit("for (int i = 0; i < _list_len_; i++) {{");
    {
      ScopedIndent _s(line_appender_);
      emit("_list_[i] = i;");
    }
    emit("}}");
  }

  void generate_listgen_kernel(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::listgen);
    const std::string glsl_kernel_name = make_kernel_name();
    emit("void {}()", glsl_kernel_name);
    this->glsl_kernel_name_ = glsl_kernel_name;
    used.listman = true;
    emit("{{ // listgen {}", stmt->snode->node_type_name);
    {
      ScopedIndent _s(line_appender_);
      if (stmt->snode->type == SNodeType::dense) {
        generate_listgen_for_dense(stmt->snode);
      } else if (stmt->snode->type == SNodeType::dynamic) {
        generate_listgen_for_dynamic(stmt->snode);
      } else {
        TI_NOT_IMPLEMENTED
      }
    }
    emit("}}\n");
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    used.buf_gtmp = true;
    emit("int {} = {};", stmt->short_name(), stmt->offset);
    ptr_signats[stmt->id] = "gtmp";
  }

  void visit(ThreadLocalPtrStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit("int {} = {};", stmt->short_name(), stmt->offset);
    ptr_signats[stmt->id] = "tls";
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
    generate_header();
    TI_ASSERT(is_top_level_);
    is_top_level_ = false;
    using Type = OffloadedStmt::TaskType;
    if (stmt->task_type == Type::serial) {
      generate_serial_kernel(stmt);
    } else if (stmt->task_type == Type::range_for) {
      generate_range_for_kernel(stmt);
    } else if (stmt->task_type == Type::struct_for) {
      generate_struct_for_kernel(stmt);
    } else if (stmt->task_type == Type::listgen) {
      generate_listgen_kernel(stmt);
    } else {
      // struct_for is automatically lowered to ranged_for for dense snodes
      // (#378). So we only need to support serial and range_for tasks.
      TI_ERROR("[glsl] Unsupported offload type={} on OpenGL arch",
               stmt->task_name());
    }
    is_top_level_ = true;
    generate_bottom();
  }

  void visit(StructForStmt *) override {
    TI_ERROR("[glsl] Struct for cannot be nested under OpenGL for now");
  }

  void visit(ClearListStmt *stmt) override {
    used.listman = true;
    emit("// clear list {}", stmt->snode->node_type_name);
    emit("_list_len_ = 0;");
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
  Kernel *get_kernel() const {
    return kernel;
  }

  std::unique_ptr<CompiledProgram> get_compiled_program() {
    // We have to set it at the last moment, to get all used feature.
    compiled_program_->set_used(used);
    return std::move(compiled_program_);
  }

  void run(const SNode &root_snode) {
    root_snode_ = &root_snode;
    root_snode_type_name_ = root_snode.node_type_name;
    kernel->ir->accept(this);
  }
};

}  // namespace

FunctionType OpenglCodeGen::gen(void) {
#if defined(TI_WITH_OPENGL)
  KernelGen codegen(kernel_, kernel_name_, struct_compiled_);
  codegen.run(*prog_->get_snode_root(SNodeTree::kFirstID));
  auto compiled = codegen.get_compiled_program();
  auto *ptr = compiled.get();
  kernel_launcher_->keep(std::move(compiled));
  return [ptr, launcher = kernel_launcher_](Context &ctx) {
    ptr->launch(ctx, launcher);
  };
#else
  TI_NOT_IMPLEMENTED
#endif
}

void OpenglCodeGen::lower() {
  auto ir = kernel_->ir.get();
  auto &config = kernel_->program->config;
  config.demote_dense_struct_fors = true;
  irpass::compile_to_executable(ir, config, kernel_,
                                /*vectorize=*/false, kernel_->grad,
                                /*ad_use_stack=*/false, config.print_ir,
                                /*lower_global_access=*/true,
                                /*make_thread_local=*/config.make_thread_local);
#ifdef _GLSL_DEBUG
  irpass::print(ir);
#endif
}

FunctionType OpenglCodeGen::compile(Program &program, Kernel &kernel) {
  this->prog_ = &program;
  this->kernel_ = &kernel;

  this->lower();
  return this->gen();
}

}  // namespace opengl
TLANG_NAMESPACE_END
