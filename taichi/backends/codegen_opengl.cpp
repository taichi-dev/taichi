#include "codegen_opengl.h"
#include <taichi/platform/opengl/opengl_api.h>

#include <string>
#include <taichi/ir.h>

TLANG_NAMESPACE_BEGIN
namespace opengl {
namespace {

std::string opengl_data_type_name(DataType dt)
{
  switch (dt) {
    case DataType::f32:
      return "float";
    case DataType::i32:
      return "int";
    case DataType::u32:
      return "uint";
    default:
      TI_NOT_IMPLEMENTED;
      break;
  }
  return "";
}

std::string opengl_unary_op_type_symbol(UnaryOpType type)
{
  switch (type)
  {
  case UnaryOpType::neg:
    return "-";
  case UnaryOpType::sqrt:
    return "sqrt";
  case UnaryOpType::floor:
    return "floor";
  case UnaryOpType::ceil:
    return "ceil";
  case UnaryOpType::abs:
    return "abs";
  case UnaryOpType::sgn:
    return "sign";
  case UnaryOpType::sin:
    return "sin";
  case UnaryOpType::asin:
    return "asin";
  case UnaryOpType::cos:
    return "cos";
  case UnaryOpType::acos:
    return "acos";
  case UnaryOpType::tan:
    return "tan";
  case UnaryOpType::tanh:
    return "tanh";
  case UnaryOpType::exp:
    return "exp";
  case UnaryOpType::log:
    return "log";
  default:
    TI_NOT_IMPLEMENTED;
  }
  return "";
}

bool is_opengl_binary_op_infix(BinaryOpType type)
{
  return !((type == BinaryOpType::min) || (type == BinaryOpType::max) ||
           (type == BinaryOpType::atan2) || (type == BinaryOpType::pow));
}

class KernelGen : public IRVisitor
{
  Kernel *kernel;

public:
  KernelGen(Kernel *kernel, std::string kernel_name)
    : kernel(kernel), kernel_name_(kernel_name),
      glsl_kernel_prefix_(kernel_name)
  {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

private: // {{{
  std::string kernel_src_code_;
  std::string indent_;
  bool is_top_level_{true};

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
    emit("");
    emit("layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;");
    emit("");
    emit("layout(std430, binding = 0) buffer data");
    emit("{{");
    emit("  int _args_[{}];", taichi_max_num_args);
    emit("  int _data_[];");
    emit("}};");
    emit("");
  }

  void generate_bottom()
  {
    // TODO: <kernel_name>() really necessary? How about just main()?
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
    emit("const uint {} = {};", stmt->raw_name(), val);
  }

  void visit(GetRootStmt *stmt) override
  {
    // Should we assert |root_stmt_| is assigned only once?
    root_stmt_ = stmt;
    emit("const uint {} = 0;", stmt->raw_name());
  }

  void visit(SNodeLookupStmt *stmt) override
  {
    std::string parent;
    if (stmt->input_snode) {
      parent = stmt->input_snode->raw_name();
    } else {
      TI_ASSERT(root_stmt_ != nullptr);
      parent = root_stmt_->raw_name();
    }

    int stride = 1; // XXX
    emit("const uint {} = {} + {} * {};",
         stmt->raw_name(), parent, stride, stmt->input_index->raw_name());
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override
  {
    emit("uint {} = ((({} + {}) >> {}) & ((1 << {}) - 1));",
         stmt->raw_name(), stmt->offset, stmt->input->raw_name(),
         stmt->bit_begin, stmt->bit_end - stmt->bit_begin);
  }

  void visit(GetChStmt *stmt) override
  {
    if (stmt->output_snode->is_place()) {
      emit("const uint {} = {} + 1 * {}; // placed",
           stmt->raw_name(), stmt->input_ptr->raw_name(), stmt->chid);
    } else {
      emit("const uint {} = {} + 1 * {};",
           stmt->raw_name(), stmt->input_ptr->raw_name(), stmt->chid);
    }
  }

  void visit(GlobalStoreStmt *stmt) override
  {
    TI_ASSERT(stmt->width() == 1);
    emit("_data_[{}] = {};", stmt->ptr->raw_name(), stmt->data->raw_name());
  }

  void visit(GlobalLoadStmt *stmt) override
  {
    TI_ASSERT(stmt->width() == 1);
    emit("{} {} = _data_[{}];", opengl_data_type_name(stmt->element_type()),
         stmt->raw_name(), stmt->ptr->raw_name());
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

  void visit(ArgLoadStmt *stmt) override {
    const auto dt = opengl_data_type_name(stmt->element_type());
    if (stmt->is_ptr) {
      emit("const {} {} = _args_[{}]; // is_ptr", dt, stmt->raw_name(), stmt->arg_id);
    } else {
      emit("const {} {} = _args_[{}];", dt, stmt->raw_name(), stmt->arg_id);
    }
  }

  void visit(ArgStoreStmt *stmt) override {
    const auto dt = metal_data_type_name(stmt->element_type());
    TI_ASSERT(!stmt->is_ptr);
    emit("_args_[{}] = {};", stmt->arg_id, stmt->val->raw_name());
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
    TI_ASSERT(is_top_level_);
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

  void run(const SNode &root_snode)
  {
    //TI_INFO("ntm:: {}", root_snode.node_type_name);
    root_snode_ = &root_snode;
    root_snode_type_name_ = root_snode.node_type_name;
    generate_header();
    irpass::print(kernel->ir);
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
    irpass::full_simplify(ir);
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

  irpass::full_simplify(ir);
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

void load_data(Context &ctx, void *data)
{
  int *data_ = (int *)data;
  for (int i = 0; i < taichi_max_num_args; i++) {
    int value = ctx.get_arg<int>(i);
    data_[i] = value;
  }
}

void save_data(Context &ctx, void *data)
{
  int *data_ = (int *)data;
  for (int i = 0; i < taichi_max_num_args; i++) {
    int value = data_[i];
    ctx.set_arg<int>(i, value);
  }
}

FunctionType OpenglCodeGen::gen(void)
{
  KernelGen codegen(kernel_, kernel_name_);
  codegen.run(*prog_->snode_root);
  const std::string kernel_source_code = codegen.kernel_source_code();
  TI_INFO("\n{}", kernel_source_code);

  return [kernel_source_code](Context &ctx) {
    void *data, *data_r;
    size_t data_size = 1024; // ...
    data = malloc(data_size);
    load_data(ctx, data);
    data_r = launch_glsl_kernel(kernel_source_code, data, data_size);
    free(data);
    save_data(ctx, data_r);
  };
}

FunctionType OpenglCodeGen::compile(Program &program, Kernel &kernel)
{
  TI_WARN("OpenGL backend currently WIP, MAY NOT WORK");
  this->prog_ = &program;
  this->kernel_ = &kernel;

  this->lower();
  return this->gen();
}

} // namespace opengl
TLANG_NAMESPACE_END
