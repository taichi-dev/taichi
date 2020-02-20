#include "codegen_opengl.h"
#include <taichi/platform/opengl/opengl_api.h>
#include <taichi/platform/opengl/opengl_kernel.h>
#include <taichi/platform/opengl/opengl_data_types.h>

#include <string>
#include <taichi/ir.h>

TLANG_NAMESPACE_BEGIN
namespace opengl {
namespace {

class KernelGen : public IRVisitor
{
  Kernel *kernel;

public:
  KernelGen(Kernel *kernel, std::string kernel_name,
      const StructCompiledResult *struct_compiled)
    : kernel(kernel),
      struct_compiled_(struct_compiled),
      kernel_name_(kernel_name),
      glsl_kernel_prefix_(kernel_name)
  {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

private: // {{{
  std::string kernel_src_code_;
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
  {
    emit("#version 430 core");
    emit("#extension GL_ARB_compute_shader: enable");
    emit("{}", struct_compiled_->source_code);
    emit("layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;");
    emit("#define NARGS {}", taichi_max_num_args);
    emit("layout(std430, binding = 0) buffer args_i32");
    emit("{{");
    emit("  int _args_i32_[NARGS * 2];");
    emit("}};");
    emit("layout(std430, binding = 0) buffer args_f32");
    emit("{{");
    emit("  float _args_f32_[NARGS * 2];");
    emit("}};");
    emit("layout(std430, binding = 0) buffer args_f64");
    emit("{{");
    emit("  double _args_f64_[NARGS];");
    emit("}};");
    emit("layout(std430, binding = 1) buffer data_i32");
    emit("{{");
    emit("  int _data_i32_[];");
    emit("}};");
    emit("layout(std430, binding = 1) buffer data_f32");
    emit("{{");
    emit("  float _data_f32_[];");
    emit("}};");
    emit("layout(std430, binding = 1) buffer data_f64");
    emit("{{");
    emit("  double _data_f64_[];");
    emit("}};");
    emit("#define _arg_i32(x) _args_i32_[(x) << 1]"); // skip to 64bit stride
    emit("#define _arg_f32(x) _args_f32_[(x) << 1]");
    emit("#define _arg_i64(x) _args_i64_[(x) << 0]");
    emit("#define _arg_f64(x) _args_f64_[(x) << 0]");
    emit("#define _mem_i32(x) _data_i32_[(x) >> 2]");
    emit("#define _mem_f32(x) _data_f32_[(x) >> 2]");
    emit("#define _mem_i64(x) _data_i64_[(x) >> 3]");
    emit("#define _mem_f64(x) _data_f64_[(x) >> 3]");
    emit("");
  }

  void generate_bottom()
  {
    // TODO(archibate): <kernel_name>() really necessary? How about just main()?
    emit("void main()");
    emit("{{");
    emit("  {}();", glsl_kernel_name_);
    emit("}}");
  }

  void visit(Block *stmt) override
  {
    if (!is_top_level_) push_indent();
    for (auto &s : stmt->statements) {
      //TI_INFO("visiting sub stmt {}", typeid(*s).name());
      s->accept(this);
    }
    if (!is_top_level_) pop_indent();
  }

  void visit(LinearizeStmt *stmt) override
  {
    std::string val = "0";
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      val = fmt::format("({} * {} + {})", val, stmt->strides[i],
                        stmt->inputs[i]->raw_name());
    }
    emit("const int {} = {};", stmt->raw_name(), val);
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
    if (stmt->output_snode->is_place()) {
      emit("{} /* place {} */ {} = {}_get{}({});",
          stmt->output_snode->node_type_name,
          opengl_data_type_name(stmt->output_snode->dt),
          stmt->raw_name(), stmt->input_snode->node_type_name,
          stmt->chid, stmt->input_ptr->raw_name());
    } else {
      emit("{} {} = {}_get{}({});", stmt->output_snode->node_type_name,
          stmt->raw_name(), stmt->input_snode->node_type_name,
          stmt->chid, stmt->input_ptr->raw_name());
    }
  }

  void visit(GlobalStoreStmt *stmt) override
  {
    TI_ASSERT(stmt->width() == 1);
    emit("_mem_{}({}) = {};", data_type_short_name(stmt->element_type()),
        stmt->ptr->raw_name(), stmt->data->raw_name());
  }

  void visit(GlobalLoadStmt *stmt) override
  {
    TI_ASSERT(stmt->width() == 1);
    emit("{} {} = _mem_{}({});", opengl_data_type_name(stmt->element_type()),
         stmt->raw_name(), data_type_short_name(stmt->element_type()), stmt->ptr->raw_name());
  }

  void visit(UnaryOpStmt *stmt) override
  {
    if (stmt->op_type != UnaryOpType::cast) {
      emit("const {} {} = {}({});", opengl_data_type_name(stmt->element_type()),
           stmt->raw_name(), opengl_unary_op_type_symbol(stmt->op_type),
           stmt->operand->raw_name());
    } else {
      // cast
      if (stmt->cast_by_value) {
        emit("const {} {} = {}({});",
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
      if (is_integral(bin->element_type())) {
        emit("const {} {} = int(floor({} / {}));", dt_name, bin_name, lhs_name,
             rhs_name);
      } else {
        emit("const {} {} = floor({} / {});", dt_name, bin_name, lhs_name,
             rhs_name);
      }
      return;
    }
    const auto binop = binary_op_type_symbol(bin->op_type);
    if (is_opengl_binary_op_infix(bin->op_type)) {
      emit("const {} {} = ({} {} {});", dt_name, bin_name, lhs_name, binop,
           rhs_name);
    } else {
      // This is a function call
      emit("const {} {} = {}({}, {});", dt_name, bin_name, binop, lhs_name,
           rhs_name);
    }
  }

  void visit(TernaryOpStmt *tri) override
  {
    TI_ASSERT(tri->op_type == TernaryOpType::select);
    emit("const {} {} = ({}) ? ({}) : ({});",
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
      emit("const {} {}({});", opengl_data_type_name(stmt->element_type()),
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
    emit("{} {}(0);",
        opengl_data_type_name(alloca->element_type()),
        alloca->raw_name());
  }

  void visit(ConstStmt *const_stmt) override
  {
    TI_ASSERT(const_stmt->width() == 1);
    emit("const {} {} = {};", opengl_data_type_name(const_stmt->element_type()),
         const_stmt->raw_name(), const_stmt->val[0].stringify());
  }

  void visit(ArgLoadStmt *stmt) override
  {
    const auto dt = opengl_data_type_name(stmt->element_type());
    if (stmt->is_ptr) {
      emit("const {} {} = _arg_{}({}); // is_ptr", dt, stmt->raw_name(),
          data_type_short_name(stmt->element_type()), stmt->arg_id);
    } else {
      emit("const {} {} = _arg_{}({});", dt, stmt->raw_name(),
          data_type_short_name(stmt->element_type()), stmt->arg_id);
    }
  }

  void visit(ArgStoreStmt *stmt) override
  {
    TI_ASSERT(!stmt->is_ptr);
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


  void visit(OffloadedStmt *stmt) override
  {
    TI_ASSERT(is_top_level_); // TODO(archibate): remove for nested kernel (?)
    is_top_level_ = false;
    using Type = OffloadedStmt::TaskType;
    if (stmt->task_type == Type::serial) {
      generate_serial_kernel(stmt);
    /*} else if (stmt->task_type == Type::range_for) {
      generate_range_for_kernel(stmt);*/
    } else {
      // struct_for is automatically lowered to ranged_for for dense snodes
      // (#378). So we only need to support serial and range_for tasks.
      TI_ERROR("Unsupported offload type={} on OpenGL arch", stmt->task_name());
    }
    is_top_level_ = true;
  }


public:
  const std::string &kernel_source_code() const
  {
    return kernel_src_code_;
  }

  SSBO *create_root_ssbo()
  {
    static SSBO *root_ssbo;
    if (!root_ssbo) {
      TI_INFO("[glsl] creating root buffer of size {} B", struct_compiled_->root_size);
      root_ssbo = new SSBO(struct_compiled_->root_size);
    }
    return root_ssbo;
  }

  void run(const SNode &root_snode)
  {
    //TI_INFO("ntm:: {}", root_snode.node_type_name);
    root_snode_ = &root_snode;
    root_snode_type_name_ = root_snode.node_type_name;
    generate_header();
    //irpass::print(kernel->ir);
    kernel->ir->accept(this);
    generate_bottom();
  }
};

} // namespace

void OpenglCodeGen::lower()
{
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
}

FunctionType OpenglCodeGen::gen(void)
{
  KernelGen codegen(kernel_, kernel_name_, struct_compiled_);
  codegen.run(*prog_->snode_root);
  SSBO *root_sb = codegen.create_root_ssbo();
  const std::string kernel_source_code = codegen.kernel_source_code();
  //TI_INFO("source of kernel [{}]:\n{}", kernel_name_, kernel_source_code);

  return [kernel_source_code, root_sb](Context &ctx) {
    // TODO(archibate): find out where get_arg<uint64_t> stored, and just new SSBO(ctx)
    SSBO *arg_sb = new SSBO(taichi_max_num_args * sizeof(uint64_t));
    arg_sb->load_arguments_from(ctx);
    std::vector<IOV> iov = {*arg_sb, *root_sb};
    /*TI_INFO("data[0] = {}", ((int*)root_sb->data)[0]);
    TI_INFO("data[1] = {}", ((int*)root_sb->data)[1]);
    TI_INFO("args[0] = {}", ((uint64_t*)arg_sb->data)[0]);
    TI_INFO("args[1] = {}", ((uint64_t*)arg_sb->data)[1]);*/
    launch_glsl_kernel(kernel_source_code, iov);
    /*TI_INFO("data[0] = {}", ((int*)root_sb->data)[0]);
    TI_INFO("data[1] = {}", ((int*)root_sb->data)[1]);
    TI_INFO("args[0] = {}", ((uint64_t*)arg_sb->data)[0]);
    TI_INFO("args[1] = {}", ((uint64_t*)arg_sb->data)[1]);*/
    arg_sb->save_returns_to(ctx);
  };
}

FunctionType OpenglCodeGen::compile(Program &program, Kernel &kernel)
{
  static bool warned;
  if (!warned) {
    TI_WARN("OpenGL backend currently WIP, MAY NOT WORK");
    warned = true;
  }

  this->prog_ = &program;
  this->kernel_ = &kernel;

  this->lower();
  return this->gen();
}

} // namespace opengl
TLANG_NAMESPACE_END
