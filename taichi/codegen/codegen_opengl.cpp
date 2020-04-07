//#define _GLSL_DEBUG 1
#include "codegen_opengl.h"
#include "taichi/backends/opengl/opengl_api.h"
#include "taichi/backends/opengl/opengl_data_types.h"
#include "taichi/util/line_appender.h"
#include "taichi/util/macros.h"
#include "taichi/ir/ir.h"
#include <string>

TLANG_NAMESPACE_BEGIN
namespace opengl {
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

struct UsedFeature {
  bool random{false};
  bool argument{false};
  bool extra_arg{false};
  bool external_ptr{false};
  bool atomic_float{false};
  bool global_temp{false};
};

struct CompiledKernel : public CompiledGLSL {
  int num_groups;
  UsedFeature used;
  std::string kernel_name;

  // disscussion:
  // https://github.com/taichi-dev/taichi/pull/696#issuecomment-609332527
  CompiledKernel(CompiledKernel &&) = default;
  CompiledKernel &operator=(CompiledKernel &&) = default;

  explicit CompiledKernel(const std::string &kernel_name_,
                          const std::string &kernel_source_code,
                          int num_groups_,
                          UsedFeature used_)
      : CompiledGLSL(kernel_source_code),
        num_groups(num_groups_),
        used(used_),
        kernel_name(kernel_name_) {
#ifdef _GLSL_DEBUG
    TI_INFO("source of kernel [{}] * {}:\n{}", kernel_name, num_groups,
            kernel_source_code);
    std::ofstream(fmt::format("/tmp/{}.comp", kernel_name))
        .write(kernel_source_code.c_str(), kernel_source_code.size());
#endif
  }

  void launch() const {
    // TI_PERF();
    this->launch_glsl(num_groups);
    // TI_PERF(kernel_name.c_str(), kernel_name.size(), 107);
  }
};

struct CompiledProgram {
  std::vector<std::unique_ptr<CompiledKernel>> kernels;
  int arg_count;
  std::map<int, size_t> ext_arr_map;
  size_t gtmp_size;

  // disscussion:
  // https://github.com/taichi-dev/taichi/pull/696#issuecomment-609332527
  CompiledProgram(CompiledProgram &&) = default;
  CompiledProgram &operator=(CompiledProgram &&) = default;

  CompiledProgram(Kernel *kernel, size_t gtmp_size) : gtmp_size(gtmp_size) {
    arg_count = kernel->args.size();
    for (int i = 0; i < arg_count; i++) {
      if (kernel->args[i].is_nparray) {
        ext_arr_map[i] = kernel->args[i].size;
      }
    }
  }

  void launch(Context &ctx, GLSLLauncher *launcher) const {
    std::vector<IOV> iov;
    iov.push_back(IOV{ctx.args, arg_count * sizeof(uint64_t)});
    auto gtmp_arr = std::vector<char>(gtmp_size);
    void *gtmp_base = gtmp_arr.data();  // std::calloc(gtmp_size, 1);
    iov.push_back(IOV{gtmp_base, gtmp_size});
    std::optional<std::vector<char>> base_arr;
    std::optional<std::vector<void *>> saved_ctx_ptrs;
    // TODO: these dirty codes are introduced by #694
    if (ext_arr_map.size()) {
      iov.push_back(
          IOV{ctx.extra_args, arg_count * taichi_max_num_args * sizeof(int)});
      if (ext_arr_map.size() == 1) {  // zero-copy for only one ext_arr
        auto it = ext_arr_map.begin();
        auto extptr = (void *)ctx.args[it->first];
        ctx.args[it->first] = 0;
        iov.push_back(IOV{extptr, it->second});
      } else {
        size_t accum_size = 0;
        std::vector<void *> ptrarr;
        for (const auto &[i, size] : ext_arr_map) {
          accum_size += size;
        }
        saved_ctx_ptrs = std::make_optional<std::vector<void *>>();
        base_arr = std::make_optional<std::vector<char>>(accum_size);
        void *baseptr = base_arr->data();
        accum_size = 0;
        for (const auto &[i, size] : ext_arr_map) {
          auto ptr = (void *)ctx.args[i];
          saved_ctx_ptrs->push_back(ptr);
          std::memcpy((char *)baseptr + accum_size, ptr, size);
          ctx.args[i] = accum_size;
          accum_size += size;
        }  // concat all extptr into my baseptr
        iov.push_back(IOV{baseptr, accum_size});
      }
    }
    for (const auto &ker : kernels) {
      auto guard = launcher->create_launch_guard(iov);
      ker->launch();
    }
    if (ext_arr_map.size() > 1) {
      void *baseptr = base_arr->data();
      auto cpit = saved_ctx_ptrs->begin();
      size_t accum_size = 0;
      for (const auto &[i, size] : ext_arr_map) {
        std::memcpy(*cpit, (char *)baseptr + accum_size, size);
        accum_size += size;
        cpit++;
      }  // extract back to all extptr from my baseptr
    }
  }
};

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
  LineAppender line_appender_, line_appender_header_;
  std::unique_ptr<CompiledProgram> compiled_program_;
  UsedFeature used;

  bool is_top_level_{true};

  StructCompiledResult *struct_compiled_;
  const SNode *root_snode_;
  GetRootStmt *root_stmt_;
  std::string kernel_name_;
  std::string glsl_kernel_name_;
  std::string root_snode_type_name_;
  std::string glsl_kernel_prefix_;
  int glsl_kernel_count_{0};
  int num_threads_{1};
  int num_groups_{1};

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    line_appender_.append(std::move(f), std::move(args)...);
  }

  void generate_header() {
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
        "layout(packed, binding = 0) buffer data_i32 { int _states_[2]; int _data_i32_[]; };\n"
        "layout(packed, binding = 0) buffer data_f32 { int _unused1_[2]; float _data_f32_[]; };\n"
        "layout(packed, binding = 0) buffer data_f64 { int _unused2_[2]; double _data_f64_[]; };\n";

    if (used.argument) {
      kernel_header +=
          "layout(packed, binding = 1) buffer args_i32 { int _args_i32_[]; };\n"
          "layout(packed, binding = 1) buffer args_f32 { float _args_f32_[]; };\n"
          "layout(packed, binding = 1) buffer args_f64 { double _args_f64_[]; };\n";
    }
    if (used.global_temp) {
      kernel_header +=
          "layout(packed, binding = 2) buffer gtmp_i32 { int _gtmp_i32_[]; };\n"
          "layout(packed, binding = 2) buffer gtmp_f32 { float _gtmp_f32_[]; };\n"
          "layout(packed, binding = 2) buffer gtmp_f64 { double _gtmp_f64_[]; };\n";
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
    }
    // clang-format on
    if (used.atomic_float && !opengl_has_GL_NV_shader_atomic_float) {  // {{{
      kernel_header += (
#include "taichi/backends/opengl/shaders/atomics_data_f32.glsl.h"
      );
#ifdef _GLSL_INT64
      kernel_header += (
#include "taichi/backends/opengl/shaders/atomics_data_f64.glsl.h"
      );
#endif
      if (used.global_temp) {
        kernel_header += (
#include "taichi/backends/opengl/shaders/atomics_gtmp_f32.glsl.h"
        );
#ifdef _GLSL_INT64
        kernel_header += (
#include "taichi/backends/opengl/shaders/atomics_gtmp_f64.glsl.h"
        );
#endif
      }
      if (used.external_ptr) {
        kernel_header += (
#include "taichi/backends/opengl/shaders/atomics_extr_f32.glsl.h"
        );
#ifdef _GLSL_INT64
        kernel_header += (
#include "taichi/backends/opengl/shaders/atomics_extr_f64.glsl.h"
        );
#endif
      }
    }                   // }}}
    if (used.random) {  // TODO(archibate): random in different offloads should
                        // share rand seed? {{{
      kernel_header += (
#include "taichi/backends/opengl/shaders/random.glsl.h"
      );
    }  // }}}

    line_appender_header_.append_raw(kernel_header);

    int threads_per_group = opengl_get_threads_per_group();
    if (num_threads_ < threads_per_group)
      threads_per_group = std::max(1, num_threads_);
    else
      num_groups_ = (num_threads_ + threads_per_group - 1) / threads_per_group;
    emit(
        "layout(local_size_x = {} /* {}, {} */, local_size_y = 1, local_size_z "
        "= 1) in;",
        threads_per_group, num_groups_, num_threads_);
    std::string extensions = "";
    if (opengl_has_GL_NV_shader_atomic_float) {
      extensions += "#extension GL_NV_shader_atomic_float: enable\n";
    }
    auto kernel_src_code =
        "#version 430 core\n" + extensions + "precision highp float;\n" +
        line_appender_header_.lines() + line_appender_.lines();
    compiled_program_->kernels.push_back(std::make_unique<CompiledKernel>(
        std::move(glsl_kernel_name_), kernel_src_code, num_groups_, used));
    line_appender_header_.clear_all();
    line_appender_.clear_all();
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
         struct_compiled_->class_children_map[parent_type],
         stmt->input_index->short_name(), stmt->snode->node_type_name);
  }

  std::map<int, std::string> ptr_signats;

  void visit(GetChStmt *stmt) override {
    emit("int {} = {} + {}; // {}", stmt->short_name(),
         stmt->input_ptr->short_name(),
         struct_compiled_
             ->class_get_map[stmt->input_snode->node_type_name][stmt->chid],
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
    } else if (stmt->op_type != UnaryOpType::cast) {
      emit("{} {} = {}({}({}));", dt_name, stmt->short_name(), dt_name,
           unary_op_type_name(stmt->op_type), stmt->operand->short_name());
    } else {
      // cast
      if (stmt->cast_by_value) {
        emit("{} {} = {}({});", dt_name, stmt->short_name(),
             opengl_data_type_name(stmt->cast_type),
             stmt->operand->short_name());
      } else if (stmt->cast_type == DataType::f32 &&
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
            "{} {} = {}({} * {} >= 0 ? abs({}) / abs({}) : sign({}) * "
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
    if (dt == DataType::i32 || opengl_has_GL_NV_shader_atomic_float) {
      emit("{} {} = {}(_{}_{}_[{} >> {}], {});",
           opengl_data_type_name(stmt->val->element_type()), stmt->short_name(),
           opengl_atomic_op_type_cap_name(stmt->op_type),
           ptr_signats.at(stmt->dest->id), data_type_short_name(dt),
           stmt->dest->short_name(), opengl_data_address_shifter(dt),
           stmt->val->short_name());
    } else {
      TI_ASSERT(dt == DataType::f32 || dt == DataType::f64);
      used.atomic_float = true;
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

  void visit(ArgStoreStmt *stmt) override {
    TI_ASSERT(!stmt->is_ptr);
    used.argument = true;
    emit("_args_{}_[{} << {}] = {};",
         data_type_short_name(stmt->element_type()), stmt->arg_id,
         opengl_argument_address_shifter(stmt->element_type()),
         stmt->val->short_name());
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
      num_threads_ = end_value - begin_value;
      emit("// range known at compile time");
      emit("int _tid = int(gl_GlobalInvocationID.x);");
      emit("if (_tid >= {}) return;", num_threads_);
      emit("int _itv = {} + _tid * {};", begin_value, 1 /* stmt->step? */);
    } else {
      TI_ERROR("[glsl] non-const range_for currently unsupported under OpenGL");
    }

    stmt->body->accept(this);
    emit("}}\n");
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    used.global_temp = true;
    emit("int {} = {};", stmt->short_name(), stmt->offset);
    ptr_signats[stmt->id] = "gtmp";
  }

  void visit(LoopIndexStmt *stmt) override {
    TI_ASSERT(!stmt->is_struct_for);
    TI_ASSERT(stmt->index == 0);  // TODO: multiple indices
    emit("int {} = _itv;", stmt->short_name());
  }

  void visit(RangeForStmt *for_stmt) override {
    TI_ASSERT(for_stmt->width() == 1);
    auto *loop_var = for_stmt->loop_var;
    if (loop_var->ret_type.data_type == DataType::i32) {
      if (!for_stmt->reversed) {
        emit("for (int {}_ = {}; {}_ < {}; {}_ = {}_ + {}) {{",
             loop_var->short_name(), for_stmt->begin->short_name(),
             loop_var->short_name(), for_stmt->end->short_name(),
             loop_var->short_name(), loop_var->short_name(), 1);
        // variable named `loop_var->short_name()` is already allocated by
        // alloca
        emit("  {} = {}_;", loop_var->short_name(), loop_var->short_name());
      } else {
        // reversed for loop
        emit("for (int {}_ = {} - 1; {}_ >= {}; {}_ = {}_ - {}) {{",
             loop_var->short_name(), for_stmt->end->short_name(),
             loop_var->short_name(), for_stmt->begin->short_name(),
             loop_var->short_name(), loop_var->short_name(), 1);
        emit("  {} = {}_;", loop_var->short_name(), loop_var->short_name());
      }
    } else {
      TI_ASSERT(!for_stmt->reversed);
      const auto type_name = opengl_data_type_name(loop_var->element_type());
      emit("for ({} {} = {}; {} < {}; {} = {} + 1) {{", type_name,
           loop_var->short_name(), for_stmt->begin->short_name(),
           loop_var->short_name(), for_stmt->end->short_name(),
           loop_var->short_name(), loop_var->short_name());
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

void OpenglCodeGen::lower() {  // {{{
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

#ifdef _GLSL_DEBUG
  irpass::print(ir);
#endif
}  // }}}

FunctionType make_launch_function(std::unique_ptr<CompiledProgram> p,
                                  GLSLLauncher *launcher) {
  static std::vector<std::unique_ptr<CompiledProgram>> no_gc;
  CompiledProgram *ptr = p.get();
  no_gc.push_back(std::move(p));
  return [ptr, launcher](Context &ctx) { ptr->launch(ctx, launcher); };
}

FunctionType OpenglCodeGen::gen(void) {
#if defined(TI_WITH_OPENGL)
  KernelGen codegen(kernel_, kernel_name_, struct_compiled_,
                    global_tmps_buffer_size_);
  codegen.run(*prog_->snode_root);
  auto compiled = codegen.get_compiled_program();

  return make_launch_function(std::move(compiled), kernel_launcher_);
#else
  TI_NOT_IMPLEMENTED
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
