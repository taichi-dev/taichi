//#define _GLSL_DEBUG 1
#include "codegen_opengl.h"

#include <string>

#include "taichi/backends/opengl/opengl_api.h"
#include "taichi/backends/opengl/opengl_data_types.h"
#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/util/line_appender.h"
#include "taichi/util/macros.h"

TLANG_NAMESPACE_BEGIN
namespace opengl {

KernelParallelAttrib::KernelParallelAttrib(OffloadedStmt *stmt)
    : KernelParallelAttrib(-1) {
  const_begin = stmt->const_begin;
  const_end = stmt->const_end;
  range_begin = stmt->const_begin ? stmt->begin_value : stmt->begin_offset;
  range_end = stmt->const_end ? stmt->end_value : stmt->end_offset;
}

namespace {

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
            StructCompiledResult *struct_compiled,
            size_t gtmp_size)
      : kernel(kernel),
        compiled_program_(std::make_unique<CompiledProgram>(kernel, gtmp_size)),
        struct_compiled_(struct_compiled),
        kernel_name_(kernel_name),
        glsl_kernel_prefix_(kernel_name) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

 private:  // {{{
  std::unique_ptr<CompiledProgram> compiled_program_;

  StructCompiledResult *struct_compiled_;
  const SNode *root_snode_;
  GetRootStmt *root_stmt_;
  std::string kernel_name_;
  std::string glsl_kernel_name_;
  std::string root_snode_type_name_;
  std::string glsl_kernel_prefix_;
  int glsl_kernel_count_{0};

  bool is_top_level_{true};
  LineAppender line_appender_;
  LineAppender line_appender_header_;
  KernelParallelAttrib kpa;
  UsedFeature used;

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    line_appender_.append(std::move(f), std::move(args)...);
  }

  void generate_header() {
  }

  std::string opengl_data_type_name(DataType dt) {  // catch & forward
    if (dt == DataType::i64)
      used.int64 = true;
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
    std::string kernel_header =
      "layout(packed, binding = 6) buffer runtime { int _rand_state_; };\n";
    kernel_header +=
      "layout(packed, binding = 0) buffer data_i32 { int _data_i32_[]; };\n"
      "layout(packed, binding = 0) buffer data_f32 { float _data_f32_[]; };\n"
      "layout(packed, binding = 0) buffer data_f64 { double _data_f64_[]; };\n";
    if (used.int64)
      kernel_header += "layout(packed, binding = 0) buffer data_i64 { int64_t _data_i64_[]; };\n";

    if (used.argument) {
      kernel_header +=
          "layout(packed, binding = 1) buffer args_i32 { int _args_i32_[]; };\n"
          "layout(packed, binding = 1) buffer args_f32 { float _args_f32_[]; };\n"
          "layout(packed, binding = 1) buffer args_f64 { double _args_f64_[]; };\n";
      if (used.int64)
        kernel_header += "layout(packed, binding = 1) buffer args_i64 { int64_t _args_i64_[]; };\n";
    }
    if (used.global_temp) {
      kernel_header +=
          "layout(packed, binding = 2) buffer gtmp_i32 { int _gtmp_i32_[]; };\n"
          "layout(packed, binding = 2) buffer gtmp_f32 { float _gtmp_f32_[]; };\n"
          "layout(packed, binding = 2) buffer gtmp_f64 { double _gtmp_f64_[]; };\n";
      if (used.int64)
        kernel_header += "layout(packed, binding = 2) buffer gtmp_i64 { int64_t _gtmp_i64_[]; };\n";
    }
    if (used.extra_arg) {
      kernel_header +=
          "layout(packed, binding = 3) buffer earg_i32 { int _earg_i32_[]; };\n";
    }
    if (used.external_ptr) {
      kernel_header +=
          "layout(packed, binding = 4) buffer extr_i32 { int _extr_i32_[]; };\n"
          "layout(packed, binding = 4) buffer extr_f32 { float _extr_f32_[]; };\n"
          "layout(packed, binding = 4) buffer extr_f64 { double _extr_f64_[]; };\n";
      if (used.int64)
        kernel_header += "layout(packed, binding = 4) buffer extr_i64 { int64_t _extr_i64_[]; };\n";
    }
    // clang-format on
    if (used.simulated_atomic_float) {
      kernel_header += (
#include "taichi/backends/opengl/shaders/atomics_data_f32.glsl.h"
      );
      if (used.global_temp) {
        kernel_header += (
#include "taichi/backends/opengl/shaders/atomics_gtmp_f32.glsl.h"
        );
      }
      if (used.external_ptr) {
        kernel_header += (
#include "taichi/backends/opengl/shaders/atomics_extr_f32.glsl.h"
        );
      }
    }
    if (used.random) {  // TODO(archibate): random in different offloads should
                        // share rand seed? {{{
      kernel_header += (
#include "taichi/backends/opengl/shaders/random.glsl.h"
      );
    }  // }}}

    if (used.fast_pow) {
      kernel_header += (
#include "taichi/backends/opengl/shaders/fast_pow.glsl.h"
      );
    }

    line_appender_header_.append_raw(kernel_header);

    emit(
        "layout(local_size_x = {} /* {}, {} */, "
        "local_size_y = 1, local_size_z = 1) in;",
        kpa.threads_per_group, kpa.num_groups, kpa.num_threads);
    std::string extensions = "";
#define PER_OPENGL_EXTENSION(x) \
  if (opengl_has_##x)           \
    extensions += "#extension " #x ": enable\n";
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION
    auto kernel_src_code =
        "#version 430 core\n" + extensions + "precision highp float;\n" +
        line_appender_header_.lines() + line_appender_.lines();
    compiled_program_->add(std::move(glsl_kernel_name_), kernel_src_code,
                           std::move(kpa), used);
    line_appender_header_.clear_all();
    line_appender_.clear_all();
    kpa = KernelParallelAttrib();
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
    TI_WARN("Cannot print inside OpenGL kernel, ignored");
  }

  void visit(RandStmt *stmt) override {
    used.random = true;
    emit("{} {} = _rand_{}();", opengl_data_type_name(stmt->ret_type.data_type),
         stmt->short_name(), data_type_short_name(stmt->ret_type.data_type));
  }

  void visit(LinearizeStmt *stmt) override {
    std::string val = "0";
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      val = fmt::format("({} * {} + {})", val, stmt->strides[i],
                        stmt->inputs[i]->short_name());
    }
    emit("int {} = {};", stmt->short_name(), val);
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override {
    emit("int {} = ((({} + {}) >> {}) & ((1 << {}) - 1));", stmt->short_name(),
         stmt->offset, stmt->input->short_name(), stmt->bit_begin,
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
    auto dt = stmt->data->element_type();
    emit("_{}_{}_[{} >> {}] = {};",
         ptr_signats.at(stmt->ptr->id),  // throw out_of_range if not a pointer
         data_type_short_name(dt), stmt->ptr->short_name(),
         opengl_data_address_shifter(dt), stmt->data->short_name());
  }

  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt = stmt->element_type();
    emit("{} {} = _{}_{}_[{} >> {}];",
         opengl_data_type_name(stmt->element_type()), stmt->short_name(),
         ptr_signats.at(stmt->ptr->id), data_type_short_name(dt),
         stmt->ptr->short_name(), opengl_data_address_shifter(dt));
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
        used.extra_arg = true;
        std::string var_name = fmt::format("_s{}_{}", i, stmt->short_name());
        emit("int {} = _earg_i32_[{} * {} + {}];", var_name, arg_id,
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
    used.external_ptr = true;
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
      emit("{} {} = {}(1 / sqrt({}));", dt_name, stmt->short_name(), dt_name,
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
      if (stmt->cast_type == DataType::f32 &&
          stmt->operand->element_type() == DataType::i32) {
        emit("{} {} = intBitsToFloat({});", dt_name, stmt->short_name(),
             stmt->operand->short_name());
      } else if (stmt->cast_type == DataType::i32 &&
                 stmt->operand->element_type() == DataType::f32) {
        emit("{} {} = floatBitsToInt({});", dt_name, stmt->short_name(),
             stmt->operand->short_name());
      } else {
        TI_ERROR("unsupported reinterpret cast");
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
          DataType::f64) {  // don't know why no atan(double, double)
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
           data_type_short_name(bin->lhs->element_type()), lhs_name, rhs_name);
      used.fast_pow = true;
      return;
    }
    const auto binop = binary_op_type_symbol(bin->op_type);
    if (is_opengl_binary_op_infix(bin->op_type)) {
      if (is_opengl_binary_op_different_return_type(bin->op_type) ||
          bin->element_type() != bin->lhs->element_type() ||
          bin->element_type() != bin->rhs->element_type()) {
        emit("{} {} = {}({} {} {});", dt_name, bin_name, dt_name, lhs_name,
             binop, rhs_name);
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
    auto dt = stmt->dest->element_type();
    if (dt == DataType::i32 ||
        (opengl_has_GL_NV_shader_atomic_int64 && dt == DataType::i64) ||
        ((stmt->op_type == AtomicOpType::add ||
          stmt->op_type == AtomicOpType::sub) &&
         ((opengl_has_GL_NV_shader_atomic_float && dt == DataType::f32) ||
          (opengl_has_GL_NV_shader_atomic_float64 && dt == DataType::f64)))) {
      emit("{} {} = {}(_{}_{}_[{} >> {}], {});",
           opengl_data_type_name(stmt->val->element_type()), stmt->short_name(),
           opengl_atomic_op_type_cap_name(stmt->op_type),
           ptr_signats.at(stmt->dest->id), data_type_short_name(dt),
           stmt->dest->short_name(), opengl_data_address_shifter(dt),
           stmt->val->short_name());
    } else {
      if (dt != DataType::f32) {
        TI_ERROR(
            "unsupported atomic operation for DataType::{}, "
            "this may because your OpenGL is missing that extension, "
            "see `glewinfo` for more details",
            data_type_short_name(dt));
      }
      used.simulated_atomic_float = true;
      emit("{} {} = {}_{}_{}({} >> {}, {});",
           opengl_data_type_name(stmt->val->element_type()), stmt->short_name(),
           opengl_atomic_op_type_cap_name(stmt->op_type),
           ptr_signats.at(stmt->dest->id), data_type_short_name(dt),
           stmt->dest->short_name(), opengl_data_address_shifter(dt),
           stmt->val->short_name());
    }
  }

  void visit(TernaryOpStmt *tri) override {
    TI_ASSERT(tri->op_type == TernaryOpType::select);
    emit("{} {} = ({}) != 0 ? ({}) : ({});",
         opengl_data_type_name(tri->element_type()), tri->short_name(),
         tri->op1->short_name(), tri->op2->short_name(),
         tri->op3->short_name());
  }

  void visit(LocalLoadStmt *stmt) override {
    bool linear_index = true;
    for (int i = 0; i < (int)stmt->ptr.size(); i++) {
      if (stmt->ptr[i].offset != i) {
        linear_index = false;
      }
    }
    if (stmt->same_source() && linear_index &&
        stmt->width() == stmt->ptr[0].var->width()) {
      auto ptr = stmt->ptr[0].var;
      emit("{} {} = {};", opengl_data_type_name(stmt->element_type()),
           stmt->short_name(), ptr->short_name());
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    emit("{} = {};", stmt->ptr->short_name(), stmt->data->short_name());
  }

  void visit(AllocaStmt *alloca) override {
    emit("{} {} = 0;", opengl_data_type_name(alloca->element_type()),
         alloca->short_name());
  }

  void visit(ConstStmt *const_stmt) override {
    TI_ASSERT(const_stmt->width() == 1);
    emit("{} {} = {};", opengl_data_type_name(const_stmt->element_type()),
         const_stmt->short_name(), const_stmt->val[0].stringify());
  }

  void visit(KernelReturnStmt *stmt) override {
    used.argument = true;
    used.int64 = true;
    // TODO: consider use _rets_{}_ instead of _args_{}_
    // TODO: use stmt->ret_id instead of 0 as index
    emit("_args_{}_[0] = {};", data_type_short_name(stmt->element_type()),
         stmt->value->short_name());
  }

  void visit(ArgLoadStmt *stmt) override {
    const auto dt = opengl_data_type_name(stmt->element_type());
    used.argument = true;
    if (stmt->is_ptr) {
      emit("int {} = _args_i32_[{} << 1]; // is ext pointer {}",
           stmt->short_name(), stmt->arg_id, dt);
    } else {
      emit("{} {} = _args_{}_[{} << {}];", dt, stmt->short_name(),
           data_type_short_name(stmt->element_type()), stmt->arg_id,
           opengl_argument_address_shifter(stmt->element_type()));
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

  void generate_range_for_kernel(OffloadedStmt *stmt) {
    TI_ASSERT(stmt->task_type == OffloadedStmt::TaskType::range_for);
    const std::string glsl_kernel_name = make_kernel_name();
    emit("void {}()", glsl_kernel_name);
    this->glsl_kernel_name_ = glsl_kernel_name;
    emit("{{ // range for");

    if (stmt->const_begin && stmt->const_end) {
      ScopedIndent _s(line_appender_);
      auto begin_value = stmt->begin_value;
      auto end_value = stmt->end_value;
      if (end_value < begin_value)
        std::swap(end_value, begin_value);
      kpa = KernelParallelAttrib(end_value - begin_value);
      emit("// range known at compile time");
      emit("int _tid = int(gl_GlobalInvocationID.x);");
      emit("if (_tid >= {}) return;", end_value - begin_value);
      emit("int _itv = {} + _tid * {};", begin_value, 1 /* stmt->step? */);
      stmt->body->accept(this);
    } else {
      {
        ScopedIndent _s(line_appender_);
        emit("// range known at runtime");
        auto begin_expr = stmt->const_begin ? std::to_string(stmt->begin_value)
                                            : fmt::format("_gtmp_i32_[{} >> 2]",
                                                          stmt->begin_offset);
        auto end_expr = stmt->const_end ? std::to_string(stmt->end_value)
                                        : fmt::format("_gtmp_i32_[{} >> 2]",
                                                      stmt->end_offset);
        emit("int _tid = int(gl_GlobalInvocationID.x);");
        emit("int _beg = {}, _end = {};", begin_expr, end_expr);
        emit("int _itv = _beg + _tid;");
        emit("if (_itv >= _end) return;");
        kpa = KernelParallelAttrib(stmt);
      }
      stmt->body->accept(this);
    }

    emit("}}\n");
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    used.global_temp = true;
    emit("int {} = {};", stmt->short_name(), stmt->offset);
    ptr_signats[stmt->id] = "gtmp";
  }

  void visit(LoopIndexStmt *stmt) override {
    TI_ASSERT(stmt->index == 0);  // TODO: multiple indices
    if (stmt->loop->is<OffloadedStmt>()) {
      TI_ASSERT(stmt->loop->as<OffloadedStmt>()->task_type ==
                OffloadedStmt::TaskType::range_for);
      emit("int {} = _itv;", stmt->short_name());
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
      emit("for (int {}_ = {}; {}_ < {}; {}_ = {}_ + {}) {{", loop_var_name,
           for_stmt->begin->short_name(), loop_var_name,
           for_stmt->end->short_name(), loop_var_name, loop_var_name, 1);
      emit("  int {} = {}_;", loop_var_name, loop_var_name);
    } else {
      // reversed for loop
      emit("for (int {}_ = {} - 1; {}_ >= {}; {}_ = {}_ - {}) {{",
           loop_var_name, for_stmt->end->short_name(), loop_var_name,
           for_stmt->begin->short_name(), loop_var_name, loop_var_name, 1);
      emit("  int {} = {}_;", loop_var_name, loop_var_name);
    }
    for_stmt->body->accept(this);
    emit("}}");
  }

  void visit(WhileControlStmt *stmt) override {
    emit("if ({} == 0) break;", stmt->cond->short_name());
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

  void visit(OffloadedStmt *stmt) override {
    generate_header();
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
      TI_ERROR("[glsl] Unsupported offload type={} on OpenGL arch",
               stmt->task_name());
    }
    is_top_level_ = true;
    generate_bottom();
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
  Kernel *get_kernel() const {
    return kernel;
  }

  std::unique_ptr<CompiledProgram> get_compiled_program() {
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
  KernelGen codegen(kernel_, kernel_name_, struct_compiled_,
                    global_tmps_buffer_size_);
  codegen.run(*prog_->snode_root);
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
  auto ir = kernel_->ir;
  auto &config = kernel_->program.config;
  config.demote_dense_struct_fors = true;
  auto res =
      irpass::compile_to_offloads(ir, config,
                                  /*vectorize=*/false, kernel_->grad,
                                  /*ad_use_stack=*/false, config.print_ir,
                                  /*lower_global_access*/ true);
  global_tmps_buffer_size_ = res.total_size;
  TI_TRACE("[glsl] Global temporary buffer size {} B",
           global_tmps_buffer_size_);
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
