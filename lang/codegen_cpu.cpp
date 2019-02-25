#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "util.h"
#include "codegen_cpu.h"
#include "program.h"
#include "ir.h"

TLANG_NAMESPACE_BEGIN

/*
class TikzGen : public Visitor {
 public:
  std::string graph;
  TikzGen() : Visitor(Visitor::Order::parent_first) {
  }

  std::string expr_name(Expr expr) {
    std::string members = "";
    if (!expr) {
      TC_ERROR("expr = 0");
    }
    if (expr->members.size()) {
      members = "[";
      bool first = true;
      for (auto m : expr->members) {
        if (!first)
          members += ", ";
        members += fmt::format("{}", m->id);
        first = false;
      }
      members += "]";
    }
    return fmt::format("\"({}){}{}\"", expr->id, members,
                       expr->node_type_name());
  }

  void link(Expr a, Expr b) {
    graph += fmt::format("{} -> {}; ", expr_name(a), expr_name(b));
  }

  void visit(Expr &expr) override {
    for (auto &ch : expr->ch) {
      link(expr, ch);
    }
  }
};

void visualize_IR(std::string fn, Expr &expr) {
  TikzGen gen;
  expr.accept(gen);
  auto cmd =
      fmt::format("python3 {}/projects/taichi_lang/make_graph.py {} '{}'",
                  get_repo_dir(), fn, gen.graph);
  trash(system(cmd.c_str()));
}
*/

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
    emit("{} {}(0);", alloca->ret_data_type_name(), alloca->ident.raw_name());
  }

  void visit(RandStmt *stmt) {
    TC_ASSERT(stmt->ret_type.data_type == DataType::f32);
    emit("const auto {} = {}::rand();", stmt->raw_name(),
         stmt->ret_data_type_name());
  }

  void visit(TmpValStmt *stmt) {
    emit("const {} {} = {};", stmt->ret_data_type_name(), stmt->raw_name(),
         stmt->val->raw_name());
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
         const_stmt->value.serialize(
             [&](long double t) {
               auto data_type = const_stmt->ret_type.data_type;
               if (data_type == DataType::f32)
                 return fmt::format("{}", (float32)t);
               else if (data_type == DataType::i32)
                 return fmt::format("{}", (int32)t);
             },
             "{"));
  }

  void visit(WhileControlStmt *stmt) {
    emit("{} = land({}, {});", stmt->mask.raw_name(), stmt->mask.raw_name(),
         stmt->cond->raw_name());
    emit("if (!any({})) break;", stmt->mask.raw_name());
  }

  void visit(WhileStmt *stmt) {
    emit("while (1) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(RangeForStmt *for_stmt) {
    auto loop_var = for_stmt->loop_var;
    if (for_stmt->parallelize) {
      //emit("#pragma omp parallel for num_threads({})", for_stmt->parallelize);
      emit("omp_set_num_threads({});", for_stmt->parallelize);
      emit("#pragma omp parallel for");
    }
    emit("for ({} {} = {}; {} < {}; {} += {}) {{",
         data_type_name(for_stmt->parent->lookup_var(loop_var).data_type),
         loop_var.raw_name(), for_stmt->begin->raw_name(), loop_var.raw_name(),
         for_stmt->end->raw_name(), loop_var.raw_name(), for_stmt->vectorize);
    for_stmt->body->accept(this);
    emit("}}");
  }

  void visit(LocalLoadStmt *stmt) {
    auto var_width = stmt->parent->lookup_var(stmt->ident).width;
    if (var_width == 1) {
      emit("const {} {}({});", stmt->ret_data_type_name(), stmt->raw_name(),
           stmt->ident.raw_name());
    } else {
      emit("const {} {}({});", stmt->ret_data_type_name(), stmt->raw_name(),
           stmt->ident.raw_name());
    }
  }

  void visit(LocalStoreStmt *stmt) {
    auto mask = stmt->parent->mask();
    if (mask) {
      emit("{} = select({}, {}, {});", stmt->ident.raw_name(), mask->raw_name(),
           stmt->stmt->raw_name(), stmt->ident.raw_name());
    } else {
      emit("{} = {};", stmt->ident.raw_name(), stmt->stmt->raw_name());
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
           stmt->snode->node_type_name, indices);
    }
  }

  void visit(GlobalStoreStmt *stmt) {
    for (int i = 0; i < stmt->data->ret_type.width; i++) {
      emit("*({} *){}[{}] = {}[{}];",
           data_type_name(stmt->data->ret_type.data_type),
           stmt->ptr->raw_name(), i, stmt->data->raw_name(), i);
    }
  }

  void visit(GlobalLoadStmt *stmt) {
    emit("{} {};", stmt->ret_data_type_name(), stmt->raw_name());
    for (int i = 0; i < stmt->ret_type.width; i++) {
      emit("{}[{}] = *{}[{}];", stmt->raw_name(), i, stmt->ptr->raw_name(), i);
    }
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
  // irpass::print(ir);
  irpass::lower(ir);
  // irpass::print(ir);
  irpass::typecheck(ir);
  // irpass::print(ir);
  irpass::loop_vectorize(ir);
  // irpass::print(ir);
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
