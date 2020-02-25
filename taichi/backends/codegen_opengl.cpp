//#define _GLSL_DEBUG 1
#include "codegen_opengl.h"
#include <taichi/platform/opengl/opengl_api.h>
#include <taichi/platform/opengl/opengl_data_types.h>

#include <string>
#include <taichi/ir.h>
#include <taichi/perf.h>

TLANG_NAMESPACE_BEGIN
namespace opengl {
namespace {

std::string opengl_atomic_op_type_cap_name(AtomicOpType type) {
  static std::map<AtomicOpType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i, s) type_names[AtomicOpType::i] = "atomic" #s;
    REGISTER_TYPE(add, Add);
    REGISTER_TYPE(sub, Sub);
    //REGISTER_TYPE(mul, Mul);
    //REGISTER_TYPE(div, Div);
    REGISTER_TYPE(max, Max);
    REGISTER_TYPE(min, Min);
    //REGISTER_TYPE(bit_and, And);
    //REGISTER_TYPE(bit_or, Or);
    //REGISTER_TYPE(bit_xor, Xor);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

struct UsedFeature
{
  bool random{false};
  bool argument{false};
  bool extra_arg{false};
  bool external_ptr{false};
  bool atomic_float{false};
  bool global_temp{false};
};

struct CompiledKernel
{
  GLProgram *glsl;
  int num_groups;
  UsedFeature used;
  std::string kernel_name;

  explicit CompiledKernel(const std::string &kernel_name_,
      const std::string &kernel_source_code,
      int num_groups_, UsedFeature used_)
    : num_groups(num_groups_), used(used_), kernel_name(kernel_name_)
  {
#ifdef _GLSL_DEBUG
    TI_INFO("source of kernel [{}] * {}:\n{}", kernel_name, num_groups, kernel_source_code);
    std::ofstream(fmt::format("/tmp/{}.comp", kernel_name))
      .write(kernel_source_code.c_str(), kernel_source_code.size());
#endif
    this->glsl = compile_glsl_program(kernel_source_code);
  }

  void launch() const
  {
    //TI_PERF();
    launch_glsl_kernel(glsl, num_groups);
    //TI_PERF(kernel_name.c_str(), kernel_name.size(), 107);
  }
};

struct CompiledProgram
{
  std::vector<CompiledKernel> kernels;
  int arg_count;
  int ext_arr_idx;
  size_t ext_arr_size;
  bool has_ext_arr{false};
  size_t gtmp_size;

  CompiledProgram(Kernel *kernel, size_t gtmp_size)
  {
    has_ext_arr = false;
    arg_count = kernel->args.size();
    for (int i = 0; i < arg_count; i++) {
      if (kernel->args[i].is_nparray) {
        if (has_ext_arr)
          TI_ERROR("[glsl] external array argument is supported to at most one in OpenGL for now");
        ext_arr_idx = i;
        ext_arr_size = kernel->args[i].size;
        has_ext_arr = true;
      }
    }
  }

  void launch(Context &ctx) const
  {
    std::vector<IOV> iov;
    iov.push_back(IOV{ctx.args, arg_count * sizeof(uint64_t)});
    void *gtmp_base = std::calloc(gtmp_size, 1);
    iov.push_back(IOV{gtmp_base, gtmp_size});
    if (has_ext_arr) {
      iov.push_back(IOV{ctx.extra_args, arg_count * taichi_max_num_args * sizeof(int)});
      void *extptr = (void *)ctx.args[ext_arr_idx];
      ctx.args[ext_arr_idx] = 0;
      iov.push_back(IOV{extptr, ext_arr_size});
    }
    begin_glsl_kernels(iov);
    for (const auto &ker: kernels) {
      ker.launch();
    }
    end_glsl_kernels(iov);
  }
};

class KernelGen : public IRVisitor
{
  Kernel *kernel;

public:
  KernelGen(Kernel *kernel, std::string kernel_name,
      const StructCompiledResult *struct_compiled, size_t gtmp_size)
    : kernel(kernel),
      compiled_program_(kernel, gtmp_size),
      struct_compiled_(struct_compiled),
      kernel_name_(kernel_name),
      glsl_kernel_prefix_(kernel_name)
  {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

private: // {{{
  std::string kernel_src_code_;
  CompiledProgram compiled_program_;
  UsedFeature used;

  std::string indent_;
  bool is_top_level_{true};

  const StructCompiledResult *struct_compiled_;
  const SNode *root_snode_;
  GetRootStmt *root_stmt_;
  std::string kernel_name_;
  std::string glsl_kernel_name_;
  std::string root_snode_type_name_;
  std::string glsl_kernel_prefix_;
  int glsl_kernel_count_{0};
  int num_threads_{1};
  int num_groups_{1};

  void push_indent()
  {
    indent_ += "  ";
  }

  void pop_indent()
  {
    indent_.pop_back();
    indent_.pop_back();
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args)
  {
    kernel_src_code_ +=
        indent_ + fmt::format(f, std::forward<Args>(args)...) + "\n";
  }

  void generate_header()
  { // {{{
    num_threads_ = 1;
    kernel_src_code_ = struct_compiled_->source_code;
  } // }}}

  void generate_bottom()
  {
    // TODO(archibate): <kernel_name>() really necessary? How about just main()?
    emit("void main()");
    emit("{{");
    if (used.random)
      emit("  _init_rand();");
    if (glsl_kernel_name_.size())
      emit("  {}();", glsl_kernel_name_);
    emit("}}");

    std::string kernel_header =
    "layout(std430, binding = 0) buffer data_i32 { int _data_i32_[]; };\n"
    "layout(std430, binding = 0) buffer data_f32 { float _data_f32_[]; };\n"
    "layout(std430, binding = 0) buffer data_f64 { double _data_f64_[]; };\n"
    "#define _mem_i32(x) _data_i32_[(x) >> 2]\n"
    "#define _mem_f32(x) _data_f32_[(x) >> 2]\n"
    "#define _mem_f64(x) _data_f64_[(x) >> 3]\n";

    if (used.argument) {
      kernel_header +=
        "layout(std430, binding = 1) buffer args_i32 { int _args_i32_[]; };\n"
        "layout(std430, binding = 1) buffer args_f32 { float _args_f32_[]; };\n"
        "layout(std430, binding = 1) buffer args_f64 { double _args_f64_[]; };\n"
        "#define _arg_i32(x) _args_i32_[(x) << 1]\n" // skip to 64bit stride
        "#define _arg_f32(x) _args_f32_[(x) << 1]\n"
        "#define _arg_f64(x) _args_f64_[(x) << 0]\n";
    }
    if (used.global_temp) {
      kernel_header +=
        "layout(std430, binding = 2) buffer gtmp_i32 { int _gtmp_i32_[]; };\n"
        "layout(std430, binding = 2) buffer gtmp_f32 { float _gtmp_f32_[]; };\n"
        "layout(std430, binding = 2) buffer gtmp_f64 { double _gtmp_f64_[]; };\n"
        "#define _gtx_i32(x) _gtmp_i32_[(x) >> 2]\n"
        "#define _gtx_f32(x) _gtmp_f32_[(x) >> 2]\n"
        "#define _gtx_f64(x) _gtmp_f64_[(x) >> 3]\n";
    }
    if (used.extra_arg) {
      kernel_header +=
        "layout(std430, binding = 3) buffer earg_i32 { int _earg_i32_[]; };\n"
        + fmt::format("#define _extra_arg(i, j) _earg_i32_[(i) * {} + (j)]\n", taichi_max_num_indices);
    }
    if (used.external_ptr) {
      kernel_header +=
        "layout(std430, binding = 4) buffer extr_i32 { int _extr_i32_[]; };\n"
        "layout(std430, binding = 4) buffer extr_f32 { float _extr_f32_[]; };\n"
        "layout(std430, binding = 4) buffer extr_f64 { double _extr_f64_[]; };\n"
        "#define _ext_ns_i32(x) _extr_i32_[(x) >> 0]\n"
        "#define _ext_ns_f32(x) _extr_f32_[(x) >> 0]\n"
        "#define _ext_ns_f64(x) _extr_f64_[(x) >> 0]\n";
    }
    kernel_header +=
      "#define _Ax_(x) x\n"
      "#define _At_(x) _Ax_(_at_##x(x))\n";
    if (used.atomic_float) { // {{{
      kernel_header += "\
#define _Atmf_Def(Add, _f_, _o_, mem, _32, float) \
float atomic##Add##_##mem##_f##_32(int addr, float rhs) \
{ \
  int old, new, ret; \
  do { \
    old = _##mem##_i##_32(addr); \
    new = floatBitsToInt(_f_(intBitsToFloat(old) _o_ rhs)); \
  } while (old != atomicCompSwap(_Ax_(_##mem##_i##_32(addr)), old, new)); \
  return intBitsToFloat(old); \
}\n\
#define _Acma_ ,\n\
#define _Atm_(func, at, x, rhs) _Ax_(func##at(x, rhs))\n\
_Atmf_Def(Add,, +, mem, 32, float)\n\
_Atmf_Def(Sub,, -, mem, 32, float)\n\
_Atmf_Def(Max, max, _Acma_, mem, 32, float)\n\
_Atmf_Def(Min, min, _Acma_, mem, 32, float)\n\
"
#ifdef _GLSL_INT64
"\
_Atmf_Def(Add,, +, 64, double)\n\
_Atmf_Def(Sub,, -, 64, double)\n\
_Atmf_Def(Max, max, _Acma_, 64, double)\n\
_Atmf_Def(Min, min, _Acma_, 64, double)\n\
"
#endif
"\n"; // discussion: https://github.com/taichi-dev/taichi/pull/495#issuecomment-590074123
    if (used.global_temp) {
      kernel_header +=
        "_Atmf_Def(Add,, +, gtx, 32, float)\n"
        "_Atmf_Def(Sub,, -, gtx, 32, float)\n"
        "_Atmf_Def(Max, max, _Acma_, gtx, 32, float)\n"
        "_Atmf_Def(Min, min, _Acma_, gtx, 32, float)\n";
    }
    if (used.external_ptr) {
      kernel_header +=
        "_Atmf_Def(Add,, +, ext_ns, 32, float)\n"
        "_Atmf_Def(Sub,, -, ext_ns, 32, float)\n"
        "_Atmf_Def(Max, max, _Acma_, ext_ns, 32, float)\n"
        "_Atmf_Def(Min, min, _Acma_, ext_ns, 32, float)\n";
    }
    } // }}}
    if (used.random) { // TODO(archibate): random in different offloads should share rand seed? {{{
      kernel_header += "\
uvec4 _rand_;\n\
\n\
void _init_rand()\n\
{\n\
  uint i = gl_GlobalInvocationID.x;\n\
  _rand_.x = 123456789 * i * 1000000007;\n\
  _rand_.y = 362436069;\n\
  _rand_.z = 521288629;\n\
  _rand_.w = 88675123;\n\
}\n\
\n\
uint _rand_u32()\n\
{\n\
  uint t = _rand_.x ^ (_rand_.x << 11);\n\
  _rand_.xyz = _rand_.yzw;\n\
  _rand_.w = (_rand_.w ^ (_rand_.w >> 19)) ^ (t ^ (t >> 8));\n\
  return _rand_.w * 1000000007;\n\
}\n\
\n\
float _rand_f32()\n\
{\n\
  return float(_rand_u32()) * (1.0 / 4294967296.0);\n\
}\n\
\n\
double _rand_f64()\n\
{\n\
  return double(_rand_f32());\n\
}\n\
\n\
int _rand_i32()\n\
{\n\
  return int(_rand_u32());\n\
}\n\
";
    } // }}}

    kernel_src_code_ = kernel_header + kernel_src_code_;

    int threads_per_group = 1792;
    if (num_threads_ < 1792)
      threads_per_group = num_threads_;
    num_groups_ = (num_threads_ + 1791) / 1792;
    if (threads_per_group == 0)
      threads_per_group = 1;
    emit("layout(local_size_x = {}, local_size_y = 1, local_size_z = 1) in;", threads_per_group);

    kernel_src_code_ = std::string("#version 430 core\n") + kernel_src_code_;
    compiled_program_.kernels.push_back(CompiledKernel(std::move(glsl_kernel_name_),
        std::move(kernel_src_code_), num_groups_, used));
    kernel_src_code_ = "";
  }

  void visit(Block *stmt) override
  {
    if (!is_top_level_) push_indent();
    for (auto &s : stmt->statements) {
      s->accept(this);
    }
    if (!is_top_level_) pop_indent();
  }

  virtual void visit(Stmt *stmt) override
  {
    TI_ERROR("[glsl] unsupported statement type {}", typeid(*stmt).name());
  }

  void visit(PrintStmt *stmt) override
  {
    TI_WARN("Cannot print inside OpenGL kernel, ignored");
  }

  void visit(RandStmt *stmt) override
  {
    used.random = true;
    emit("{} {} = _rand_{}();", opengl_data_type_name(stmt->ret_type.data_type),
        stmt->raw_name(), data_type_short_name(stmt->ret_type.data_type));
  }

  void visit(LinearizeStmt *stmt) override
  {
    std::string val = "0";
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      val = fmt::format("({} * {} + {})", val, stmt->strides[i],
                        stmt->inputs[i]->raw_name());
    }
    emit("int {} = {};", stmt->raw_name(), val);
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override
  {
    emit("int {} = ((({} + {}) >> {}) & ((1 << {}) - 1));",
         stmt->raw_name(), stmt->offset, stmt->input->raw_name(),
         stmt->bit_begin, stmt->bit_end - stmt->bit_begin);
  }

  void visit(GetRootStmt *stmt) override
  {
    // Should we assert |root_stmt_| is assigned only once?
    root_stmt_ = stmt;
    emit("{} {} = 0;", root_snode_type_name_, stmt->raw_name());
  }

  void visit(SNodeLookupStmt *stmt) override
  {
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

    emit("{}_ch {} = {}_children({}, {});", stmt->snode->node_type_name,
         stmt->raw_name(), parent_type, parent->raw_name(),
         stmt->input_index->raw_name());
  }

  void visit(GetChStmt *stmt) override
  {
    emit("{} {} = {}_get{}({});", stmt->output_snode->node_type_name,
        stmt->raw_name(), stmt->input_snode->node_type_name,
        stmt->chid, stmt->input_ptr->raw_name());
    if (stmt->output_snode->is_place())
      // The best way I could think to distinguish root_ptr and external_ptr in GLSL
      emit("#define _at_{} _mem_{}", stmt->raw_name(),
          data_type_short_name(stmt->output_snode->dt));
  }

  void visit(GlobalStoreStmt *stmt) override
  {
    TI_ASSERT(stmt->width() == 1);
    emit("_At_({}) = {};", stmt->ptr->raw_name(), stmt->data->raw_name());
  }

  void visit(GlobalLoadStmt *stmt) override
  {
    TI_ASSERT(stmt->width() == 1);
    emit("{} {} = _At_({});", opengl_data_type_name(stmt->element_type()),
         stmt->raw_name(), stmt->ptr->raw_name());
  }

  void visit(ExternalPtrStmt *stmt) override
{
    // Used mostly for transferring data between host (e.g. numpy array) and
    // Metal.
    TI_ASSERT(stmt->width() == 1);
    const auto linear_index_name =
        fmt::format("{}_linear_index_", stmt->raw_name());
    emit("int {} = 0;", linear_index_name);
    emit("{{ // linear seek");
    push_indent();
    const auto *argload = stmt->base_ptrs[0]->as<ArgLoadStmt>();
    const int arg_id = argload->arg_id;
    const int num_indices = stmt->indices.size();
    std::vector<std::string> size_var_names;
    for (int i = 0; i < num_indices; i++) {
      used.extra_arg = true;
      std::string var_name = fmt::format("{}_size{}_", stmt->raw_name(), i);
      emit("int {} = _extra_arg({}, {});", var_name, arg_id, i);
      size_var_names.push_back(std::move(var_name));
    }
    for (int i = 0; i < num_indices; i++) {
      emit("{} *= {};", linear_index_name, size_var_names[i]);
      emit("{} += {};", linear_index_name, stmt->indices[i]->raw_name());
    }

    pop_indent();
    emit("}}");

    emit("int {} = ({} + {});", stmt->raw_name(),
         stmt->base_ptrs[0]->raw_name(), linear_index_name);
      used.external_ptr = true;
      emit("#define _at_{} _ext_ns_{}", stmt->raw_name(),
          data_type_short_name(stmt->element_type()));
  }

  void visit(UnaryOpStmt *stmt) override
  {
    if (stmt->op_type != UnaryOpType::cast) {
      emit("{} {} = {}({}({}));", opengl_data_type_name(stmt->element_type()),
           stmt->raw_name(), opengl_data_type_name(stmt->element_type()),
           opengl_unary_op_type_symbol(stmt->op_type), stmt->operand->raw_name());
    } else {
      // cast
      if (stmt->cast_by_value) {
        emit("{} {} = {}({});",
             opengl_data_type_name(stmt->element_type()), stmt->raw_name(),
             opengl_data_type_name(stmt->cast_type), stmt->operand->raw_name());
      } else {
        TI_NOT_IMPLEMENTED;
      }
    }
  }

  void visit(BinaryOpStmt *bin) override
  {
    const auto dt_name = opengl_data_type_name(bin->element_type());
    const auto lhs_name = bin->lhs->raw_name();
    const auto rhs_name = bin->rhs->raw_name();
    const auto bin_name = bin->raw_name();
    if (bin->op_type == BinaryOpType::floordiv) {
      // TODO(archibate): add a lossless version for (int // int)
      // NOTE: the 1e-6 here is for precision reason, or `7 // 7` will obtain 0 instead of 1
      emit("{} {} = {}(floor((float({}) * (1 + sign({} * {}) * 1e-6)) / float({})));",
          dt_name, bin_name, dt_name, lhs_name, lhs_name, rhs_name, rhs_name);
      return;
    } else if (bin->op_type == BinaryOpType::mod) {
      // NOTE: the GLSL built-in function `mod()` is a pythonic mod: x - y * floor(x / y)
      emit("{} {} = {} - {} * int({} / {});", dt_name, bin_name, lhs_name, rhs_name,
          lhs_name, rhs_name);
      return;
    }
    const auto binop = binary_op_type_symbol(bin->op_type);
    if (is_opengl_binary_op_infix(bin->op_type)) {
      emit("{} {} = {}({} {} {});", dt_name, bin_name, dt_name,
          lhs_name, binop, rhs_name);
    } else {
      // This is a function call
      emit("{} {} = {}({}, {});", dt_name, bin_name, binop, lhs_name,
          rhs_name);
    }
  }

  void visit(AtomicOpStmt *stmt) override
  {
    TI_ASSERT(stmt->width() == 1);
    if (stmt->val->element_type() == DataType::i32
        || stmt->val->element_type() == DataType::u32) {
      emit("{} {} = {}(_At_({}), {});", opengl_data_type_name(stmt->val->element_type()),
          stmt->raw_name(), opengl_atomic_op_type_cap_name(stmt->op_type),
          stmt->dest->raw_name(), stmt->val->raw_name());
    } else {
      TI_ASSERT(stmt->val->element_type() == DataType::f32
        || stmt->val->element_type() == DataType::f64);
      used.atomic_float = true;
      emit("{} {} = _Atm_({}, _at_{}, {}, {});", opengl_data_type_name(stmt->val->element_type()),
          stmt->raw_name(), opengl_atomic_op_type_cap_name(stmt->op_type),
          stmt->dest->raw_name(), stmt->dest->raw_name(), stmt->val->raw_name());
    }
  }

  void visit(TernaryOpStmt *tri) override
  {
    TI_ASSERT(tri->op_type == TernaryOpType::select);
    emit("{} {} = ({}) != 0 ? ({}) : ({});",
         opengl_data_type_name(tri->element_type()), tri->raw_name(),
         tri->op1->raw_name(), tri->op2->raw_name(), tri->op3->raw_name());
  }

  void visit(LocalLoadStmt *stmt) override
  {
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
           stmt->raw_name(), ptr->raw_name());
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(LocalStoreStmt *stmt) override
  {
    emit("{} = {};", stmt->ptr->raw_name(), stmt->data->raw_name());
  }

  void visit(AllocaStmt *alloca) override
  {
    emit("{} {} = 0;",
        opengl_data_type_name(alloca->element_type()),
        alloca->raw_name());
  }

  void visit(ConstStmt *const_stmt) override
  {
    TI_ASSERT(const_stmt->width() == 1);
    emit("{} {} = {};", opengl_data_type_name(const_stmt->element_type()),
         const_stmt->raw_name(), const_stmt->val[0].stringify());
  }

  void visit(ArgLoadStmt *stmt) override
  {
    const auto dt = opengl_data_type_name(stmt->element_type());
    used.argument = true;
    if (stmt->is_ptr) {
      emit("int {} = _arg_i32({}); // is ext pointer {}", stmt->raw_name(), stmt->arg_id, dt);
    } else {
      emit("{} {} = _arg_{}({});", dt, stmt->raw_name(),
          data_type_short_name(stmt->element_type()), stmt->arg_id);
    }
  }

  void visit(ArgStoreStmt *stmt) override
  {
    TI_ASSERT(!stmt->is_ptr);
    used.argument = true;
    emit("_arg_{}({}) = {};", data_type_short_name(stmt->element_type()),
        stmt->arg_id, stmt->val->raw_name());
  }

  std::string make_kernel_name()
  {
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

    push_indent();
    if (stmt->const_begin && stmt->const_end) {
      auto begin_value = stmt->begin_value;
      auto end_value = stmt->end_value;
      if (end_value < begin_value)
        std::swap(end_value, begin_value);
      num_threads_ = end_value - begin_value;
      emit("// range known at compile time");
      emit("int _thread_id_ = int(gl_GlobalInvocationID.x);");
      emit("if (_thread_id_ >= {}) return;", num_threads_);
      emit("int _it_value_ = {} + _thread_id_ * {};",
          begin_value, 1 /* stmt->step? */);
    } else {
      TI_ERROR("[glsl] non-const range_for currently unsupported under OpenGL");
      /*range_for_attribs.begin =
          (stmt->const_begin ? stmt->begin_value : stmt->begin_offset);
      range_for_attribs.end =
          (stmt->const_end ? stmt->end_value : stmt->end_offset);*/
    }
    pop_indent();

    stmt->body->accept(this);
    emit("}}\n");
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    used.global_temp = true;
    emit("int {} = {};", stmt->raw_name(), stmt->offset);
    emit("#define _at_{} _gtx_{}", stmt->raw_name(),
        data_type_short_name(stmt->element_type()));
  }

  void visit(LoopIndexStmt *stmt) override
  {
    TI_ASSERT(!stmt->is_struct_for);
    TI_ASSERT(stmt->index == 0); // TODO: multiple indices
    emit("int {} = _it_value_;", stmt->raw_name());
  }

  void visit(RangeForStmt *for_stmt) override
  {
    TI_ASSERT(for_stmt->width() == 1);
    auto *loop_var = for_stmt->loop_var;
    if (loop_var->ret_type.data_type == DataType::i32) {
      if (!for_stmt->reversed) {
        emit("for (int {}_ = {}; {}_ < {}; {}_ = {}_ + {}) {{",
             loop_var->raw_name(), for_stmt->begin->raw_name(),
             loop_var->raw_name(), for_stmt->end->raw_name(),
             loop_var->raw_name(), loop_var->raw_name(), 1);
        // variable named `loop_var->raw_name()` is already allocated by alloca
        emit("  {} = {}_;", loop_var->raw_name(), loop_var->raw_name());
      } else {
        // reversed for loop
        emit("for (int {}_ = {} - 1; {}_ >= {}; {}_ = {}_ - {}) {{",
             loop_var->raw_name(), for_stmt->end->raw_name(),
             loop_var->raw_name(), for_stmt->begin->raw_name(),
             loop_var->raw_name(), loop_var->raw_name(), 1);
        emit("  {} = {}_;", loop_var->raw_name(), loop_var->raw_name());
      }
    } else {
      TI_ASSERT(!for_stmt->reversed);
      const auto type_name = opengl_data_type_name(loop_var->element_type());
      emit("for ({} {} = {}; {} < {}; {} = {} + 1) {{", type_name,
           loop_var->raw_name(), for_stmt->begin->raw_name(),
           loop_var->raw_name(), for_stmt->end->raw_name(),
           loop_var->raw_name(), loop_var->raw_name());
    }
    for_stmt->body->accept(this);
    emit("}}");
  }

  void visit(WhileControlStmt *stmt) override
  {
    emit("if ({} == 0) break;", stmt->cond->raw_name());
  }

  void visit(WhileStmt *stmt) override
  {
    emit("while (true) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(OffloadedStmt *stmt) override
  {
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
      TI_ERROR("[glsl] Unsupported offload type={} on OpenGL arch", stmt->task_name());
    }
    is_top_level_ = true;
    generate_bottom();
  }

  void visit(StructForStmt *) override
  {
    TI_ERROR("[glsl] Struct for cannot be nested under OpenGL for now");
  }

  void visit(IfStmt *if_stmt) override {
    emit("if ({} != 0) {{", if_stmt->cond->raw_name());
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
  Kernel *get_kernel() const
  {
    return kernel;
  }

  CompiledProgram get_compiled_program() const
  {
    return compiled_program_;
  }

  void run(const SNode &root_snode)
  {
    root_snode_ = &root_snode;
    root_snode_type_name_ = root_snode.node_type_name;
    kernel->ir->accept(this);
  }
};

} // namespace

void OpenglCodeGen::lower()
{ // {{{
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
} // }}}

FunctionType OpenglCodeGen::gen(void)
{
  KernelGen codegen(kernel_, kernel_name_, struct_compiled_, global_tmps_buffer_size_);
  codegen.run(*prog_->snode_root);
  auto compiled = codegen.get_compiled_program();
  return [compiled](Context &ctx) {
    compiled.launch(ctx);
  };
}

FunctionType OpenglCodeGen::compile(Program &program, Kernel &kernel)
{
  static bool warned;
  if (!warned) {
    TI_WARN("[glsl] OpenGL backend currently WIP, MAY NOT WORK");
    warned = true;
  }

  this->prog_ = &program;
  this->kernel_ = &kernel;

  this->lower();
  return this->gen();
}

} // namespace opengl
TLANG_NAMESPACE_END
