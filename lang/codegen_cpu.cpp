#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "util.h"
#include "codegen_cpu.h"
#include "program.h"
#include "optimizer.h"
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

  void visit(BinaryOpStmt *bin) {
    emit("const {} {} = {}({}, {});", bin->ret_data_type_name(),
         bin->raw_name(), binary_type_name(bin->op_type), bin->lhs->raw_name(),
         bin->rhs->raw_name());
  }

  void visit(IfStmt *if_stmt) {
    emit("if ({}) {{", if_stmt->cond->raw_name());
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      emit("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    emit("}}");
  }

  void visit(PrintStmt *print_stmt) {
    emit("std::cout << {} << std::endl;", print_stmt->stmt->raw_name());
  }

  void visit(ConstStmt *const_stmt) {
    emit("const {} {}({});",  // const_stmt->ret_data_type_name(),
         const_stmt->ret_type.str(), const_stmt->raw_name(), const_stmt->value);
  }

  void visit(RangeForStmt *for_stmt) {
    auto loop_var = for_stmt->loop_var;
    emit("for ({} {} = {}; {} < {}; {}++) {{",
         data_type_name(for_stmt->parent->lookup_var(loop_var).data_type),
         loop_var.raw_name(), for_stmt->begin->raw_name(), loop_var.raw_name(),
         for_stmt->end->raw_name(), loop_var.raw_name());
    for_stmt->body->accept(this);
    emit("}}");
  }

  void visit(LocalLoadStmt *stmt) {
    emit("const {} {} = {};", stmt->ret_data_type_name(), stmt->raw_name(),
         stmt->ident.raw_name());
  }

  void visit(LocalStoreStmt *stmt) {
    emit("{} = {};", stmt->ident.raw_name(), stmt->stmt->raw_name());
  }

  void visit(GlobalPtrStmt *stmt) {
    std::string indices = "(root, ";
    for (int i = 0; i < max_num_indices; i++) {
      if (i < (int)stmt->indices.size()) {
        indices += stmt->indices[i]->raw_name();
      } else {
        indices += "0";
      }
      if (i + 1 < max_num_indices)
        indices += ",";
    }
    indices += ")";
    emit("void *{} = access_{}{};", stmt->raw_name(),
         stmt->snode->node_type_name, indices);
  }

  void visit(GlobalStoreStmt *stmt) {
    emit("*({} *){} = {};", stmt->data->ret_data_type_name(),
         stmt->ptr->raw_name(), stmt->data->raw_name());
  }

  void visit(GlobalLoadStmt *stmt) {
    emit("auto {} = *({} *){};", stmt->raw_name(), stmt->ret_data_type_name(),
         stmt->ptr->raw_name());
  }
};

void CPUCodeGen::codegen(Kernel &kernel) {
  // TC_ASSERT(mode == Mode::vector);
  this->prog = &kernel.program;
  this->current_kernel = &kernel;
  this->simd_width = prog->config.simd_width;
  this->num_groups = kernel.parallel_instances;

  has_residual = false;

  {
    CODE_REGION(header);
    generate_header();
  }

  IRCodeGen::run(this, kernel.ir);

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
