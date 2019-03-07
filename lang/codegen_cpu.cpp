#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "util.h"
#include "codegen_cpu.h"
#include "program.h"
#include "ir.h"

TLANG_NAMESPACE_BEGIN

class IRCodeGen : public IRVisitor {
 public:
  CodeGenBase *codegen;
  IRCodeGen(CodeGenBase *codegen) : codegen(codegen) {
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    codegen->emit(f, std::forward<Args>(args)...);
  }

  static void run(CodeGenBase *codegen, IRNode *node) {
    auto p = IRCodeGen(codegen);
    node->accept(&p);
  }

  void visit(Block *stmt_list) {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(AllocaStmt *alloca) {
    emit("{} {}(0);", alloca->ret_data_type_name(), alloca->raw_name());
  }

  void visit(RandStmt *stmt) {
    TC_ASSERT(stmt->ret_type.data_type == DataType::f32);
    emit("const auto {} = {}::rand();", stmt->raw_name(),
         stmt->ret_data_type_name());
  }

  void visit(BinaryOpStmt *bin) {
    emit("const {} {} = {}({}, {});", bin->ret_data_type_name(),
         bin->raw_name(), binary_type_name(bin->op_type), bin->lhs->raw_name(),
         bin->rhs->raw_name());
  }

  void visit(UnaryOpStmt *stmt) {
    if (stmt->op_type != UnaryType::cast) {
      emit("const {} {} = {}({});", stmt->ret_data_type_name(),
           stmt->raw_name(), unary_type_name(stmt->op_type),
           stmt->rhs->raw_name());
    } else {
      emit("const {} {} = cast<{}>({});", stmt->ret_data_type_name(),
           stmt->raw_name(), data_type_name(stmt->cast_type),
           stmt->rhs->raw_name());
    }
  }

  void visit(IfStmt *if_stmt) {
    // emit("if ({}) {{", if_stmt->cond->raw_name());
    emit("{{");
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      // emit("}} else {{");
      emit("}}  {{");
      if_stmt->false_statements->accept(this);
    }
    emit("}}");
  }

  void visit(PrintStmt *print_stmt) {
    emit("std::cout << \"[debug] \" \"{}\" \" = \" << {} << std::endl;",
         print_stmt->str, print_stmt->stmt->raw_name());
  }

  void visit(ConstStmt *const_stmt) {
    emit("const {} {}({});", const_stmt->ret_type.str(), const_stmt->raw_name(),
         const_stmt->val.serialize(
             [&](const TypedConstant &t) { return t.stringify(); }, "{"));
  }

  void visit(WhileControlStmt *stmt) {
    emit("{} = bit_and({}, {});", stmt->mask->raw_name(),
         stmt->mask->raw_name(), stmt->cond->raw_name());
    emit("if (!any({})) break;", stmt->mask->raw_name());
  }

  void visit(WhileStmt *stmt) {
    emit("while (1) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(RangeForStmt *for_stmt) {
    auto loop_var = for_stmt->loop_var;
    if (for_stmt->parallelize) {
      // emit("#pragma omp parallel for num_threads({})",
      // for_stmt->parallelize);
      emit("omp_set_num_threads({});", for_stmt->parallelize);
      emit("#pragma omp parallel for private({})", loop_var->raw_name());
    }
    if (loop_var->ret_type.width == 1 &&
        loop_var->ret_type.data_type == DataType::i32) {
      emit("for (int {}_ = {}; {}_ < {}; {}_ = {}_ + {}) {{",
           loop_var->raw_name(), for_stmt->begin->raw_name(),
           loop_var->raw_name(), for_stmt->end->raw_name(),
           loop_var->raw_name(), loop_var->raw_name(), for_stmt->vectorize);
      emit("{} = {}_;", loop_var->raw_name(), loop_var->raw_name());
    } else {
      emit("for ({} {} = {}; {} < {}; {} = {} + {}({})) {{",
           loop_var->ret_data_type_name(), loop_var->raw_name(),
           for_stmt->begin->raw_name(), loop_var->raw_name(),
           for_stmt->end->raw_name(), loop_var->raw_name(),
           loop_var->raw_name(), loop_var->ret_data_type_name(),
           for_stmt->vectorize);
    }
    for_stmt->body->accept(this);
    emit("}}");
  }

  void visit(LocalLoadStmt *stmt) {
    // TODO: optimize for partially vectorized load...

    bool linear_index = true;
    for (int i = 0; i < (int)stmt->ptr.size(); i++) {
      if (stmt->ptr[i].offset != i) {
        linear_index = false;
      }
    }
    if (stmt->same_source() && linear_index &&
        stmt->width() == stmt->ptr[0].var->width()) {
      auto ptr = stmt->ptr[0].var;
      emit("const {} {}({});", stmt->ret_data_type_name(), stmt->raw_name(),
           ptr->raw_name());
    } else {
      std::string init_v;
      for (int i = 0; i < stmt->width(); i++) {
        init_v += fmt::format("{}[{}]", stmt->ptr[i].var->raw_name(),
                              stmt->ptr[i].offset);
        if (i + 1 < stmt->width()) {
          init_v += ", ";
        }
      }
      emit("const {} {}({{{}}});", stmt->ret_data_type_name(), stmt->raw_name(),
           init_v);
    }
  }

  void visit(LocalStoreStmt *stmt) {
    auto mask = stmt->parent->mask();
    if (mask) {
      emit("{} = select({}, {}, {});", stmt->ident->raw_name(),
           mask->raw_name(), stmt->stmt->raw_name(), stmt->ident->raw_name());
    } else {
      TC_P(stmt->ident);
      TC_P(stmt->stmt);
      emit("{} = {};", stmt->ident->raw_name(), stmt->stmt->raw_name());
    }
  }

  void visit(GlobalPtrStmt *stmt) {
    emit("{} *{}[{}];", data_type_name(stmt->ret_type.data_type),
         stmt->raw_name(), stmt->ret_type.width);
    for (int l = 0; l < stmt->ret_type.width; l++) {
      std::string indices = "(root, ";
      for (int i = 0; i < max_num_indices; i++) {
        if (i < (int)stmt->indices.size()) {
          indices += stmt->indices[i]->raw_name() + fmt::format("[{}]", l);
        } else {
          indices += "0";
        }
        if (i + 1 < max_num_indices)
          indices += ",";
      }
      indices += ")";
      emit("{}[{}] = access_{}{};", stmt->raw_name(), l,
           stmt->snode[l]->node_type_name, indices);
    }
  }

  void visit(GlobalStoreStmt *stmt) {
    if (!current_program->config.force_vectorized_global_store) {
      for (int i = 0; i < stmt->data->ret_type.width; i++) {
        emit("*({} *){}[{}] = {}[{}];",
             data_type_name(stmt->data->ret_type.data_type),
             stmt->ptr->raw_name(), i, stmt->data->raw_name(), i);
      }
    } else {
      emit("{}.store({}[0]);", stmt->data->raw_name(), stmt->ptr->raw_name());
    }
  }

  void visit(GlobalLoadStmt *stmt) {
    if (!current_program->config.force_vectorized_global_load) {
      emit("{} {};", stmt->ret_data_type_name(), stmt->raw_name());
      for (int i = 0; i < stmt->ret_type.width; i++) {
        emit("{}[{}] = *{}[{}];", stmt->raw_name(), i, stmt->ptr->raw_name(),
             i);
      }
    } else {
      emit("const auto {} = {}::load({}[0]);", stmt->raw_name(),
           stmt->ret_data_type_name(), stmt->ptr->raw_name());
    }
  }

  void visit(ElementShuffleStmt *stmt) {
    emit("const {} {}({});", stmt->ret_data_type_name(), stmt->raw_name(),
         stmt->elements.serialize(
             [](const VectorElement &elem) {
               return fmt::format("{}[{}]", elem.stmt->raw_name(), elem.index);
             },
             "{"));
  }
};

void CPUCodeGen::codegen(Kernel &kernel) {
  // TC_ASSERT(mode == Mode::vector);
  this->prog = &kernel.program;
  this->current_kernel = &kernel;

  {
    CODE_REGION(header);
    generate_header();
  }

  auto ir = kernel.ir;
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  irpass::lower(ir);
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  irpass::typecheck(ir);
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  irpass::slp_vectorize(ir);
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  irpass::loop_vectorize(ir);
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  IRCodeGen::run(this, ir);

  {
    CODE_REGION(tail);
    code_suffix = "";
    generate_tail();
  }
}

FunctionType CPUCodeGen::get(Program &prog, Kernel &kernel) {
  // auto mode = CPUCodeGen::Mode::vv;
  auto mode = CPUCodeGen::Mode::intrinsics;
  auto simd_width = prog.config.simd_width;
  this->mode = mode;
  this->simd_width = simd_width;
  codegen(kernel);
  return compile();
}

FunctionType Program::compile(Kernel &kernel) {
  FunctionType ret = nullptr;
  if (config.arch == Arch::x86_64) {
    CPUCodeGen codegen;
    if (!kernel.name.empty()) {
      codegen.source_name = kernel.name + ".cpp";
    }
    ret = codegen.get(*this, kernel);
  } else if (config.arch == Arch::gpu) {
    TC_NOT_IMPLEMENTED
    // GPUCodeGen codegen;
    // function = codegen.get(*this);
  } else {
    TC_NOT_IMPLEMENTED;
  }
  TC_ASSERT(ret);
  return ret;
}

std::string CodeGenBase::get_source_fn() {
  return fmt::format("{}/{}/{}", get_project_fn(), folder, source_name);
}

FunctionType CPUCodeGen::compile() {
  write_code_to_file();
  auto cmd = get_current_program().config.compile_cmd(get_source_fn(),
                                                      get_library_fn());
  auto compile_ret = std::system(cmd.c_str());
  if (compile_ret != 0) {
    auto cmd = get_current_program().config.compile_cmd(get_source_fn(),
                                                        get_library_fn(), true);
    trash(std::system(cmd.c_str()));
    TC_ERROR("Source {} compilation failed.", get_source_fn());
  }
  disassemble();
  return load_function();
}

TLANG_NAMESPACE_END
