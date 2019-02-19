#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "util.h"
#include "codegen_cpu.h"
#include "slp_vectorizer.h"
#include "program.h"
#include "loop_vectorizer.h"
#include "optimizer.h"
#include "adapter_preprocessor.h"
#include "vector_splitter.h"
#include "desugaring.h"
#include "ir.h"

TLANG_NAMESPACE_BEGIN

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
    emit("{}alloca {}", alloca->type_hint(), alloca->ident.name());
  }

  void visit(BinaryOpStmt *bin) {
    emit("{}{} = {} {} {}", bin->type_hint(), bin->name(),
         binary_type_name(bin->op_type), bin->lhs->name(), bin->rhs->name());
  }

  void visit(IfStmt *if_stmt) {
    emit("if {} {{", if_stmt->cond->name());
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      emit("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    emit("}}");
  }

  void visit(PrintStmt *print_stmt) {
    emit("{}print {}", print_stmt->type_hint(), print_stmt->stmt->name());
  }

  void visit(ConstStmt *const_stmt) {
    emit("{}{} = const {}", const_stmt->type_hint(), const_stmt->name(),
         const_stmt->value);
  }

  void visit(FrontendForStmt *for_stmt) {
    emit("for {} in range({}, {}) {{", for_stmt->loop_var_id.name(),
         for_stmt->begin->serialize(), for_stmt->end->serialize());
    for_stmt->body->accept(this);
    emit("}}");
  }

  void visit(LocalLoadStmt *stmt) {
    emit("{}{} = load {}", stmt->type_hint(), stmt->name(), stmt->ident.name());
  }

  void visit(LocalStoreStmt *stmt) {
    emit("[store] {} = {}", stmt->ident.name(), stmt->stmt->name());
  }
};

void CPUCodeGen::codegen(Kernel &kernel) {
  // TC_ASSERT(mode == Mode::vector);
  this->prog = &kernel.program;
  this->current_kernel = &kernel;
  this->simd_width = prog->config.simd_width;
  this->num_groups = kernel.parallel_instances;

  auto snode = prog->current_snode;
  while (snode->type == SNodeType::forked) {
    snode = snode->parent;
  }
  has_residual = kernel.parallel_instances > 1 &&
                 (snode->type == SNodeType::indirect ||
                  snode->parent->type == SNodeType::dynamic);

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
