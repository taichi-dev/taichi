#pragma once

#include "../include/context.h"
#include "../include/unified_allocator.h"
#include "util.h"
#include "snode.h"
#include "ir.h"

TLANG_NAMESPACE_BEGIN

class Program;
extern Program *current_program;
extern SNode root;

TC_FORCE_INLINE Program &get_current_program() {
  return *current_program;
}

class Program {
 public:
  // Should be copiable
  class Kernel {
   public:
    IRNode *ir;
    Program &program;
    FunctionType compiled;
    std::string name;

    Kernel(Program &program, std::function<void()> func) : program(program) {
      context = std::make_unique<FrontendContext>();
      ir = context->root();

      program.start_function_definition(this);
      func();

      program.end_function_definition();

      compile();
    }

    void compile() {
      compiled = program.compile(*this);
    }

    void operator()() {
      auto c = program.get_context();
      compiled(c);
    }
  };

  Kernel *current_kernel;
  SNode *current_snode;
  SNode *snode_root;
  void *data_structure;
  CompileConfig config;

  std::vector<Kernel> functions;
  int index_counter;

  std::string layout_fn;

  Context get_context() {
    Context context;
    context.buffers[0] = data_structure;
    return context;
  }

  Program(Arch arch = Arch::x86_64) {
    UnifiedAllocator::create();
    TC_ASSERT(current_program == nullptr);
    current_program = this;
    config.arch = arch;
    current_kernel = nullptr;
    snode_root = nullptr;
    index_counter = 0;
  }

  ~Program() {
    current_program = nullptr;
    UnifiedAllocator::free();
  }

  void layout(std::function<void()> func) {
    root = SNode(0, SNodeType::root);
    snode_root = &root;
    func();
    materialize_layout();
  }

  Kernel def(const std::function<void()> &body) {
    // Expr::set_allow_store(true);
    auto func = Kernel(*this, body);
    functions.push_back(func);
    // Expr::set_allow_store(false);
    return func;
  }

  Kernel kernel(const std::function<void()> &body) {
    // Expr::set_allow_store(true);
    auto func = Kernel(*this, body);
    // Expr::set_allow_store(false);
    functions.push_back(func);
    current_snode = nullptr;
    return func;
  }

  void start_function_definition(Kernel *func) {
    current_kernel = func;
  }

  void end_function_definition() {
    current_kernel = nullptr;
  }

  FunctionType compile(Kernel &kernel);

  void materialize_layout();

  inline Kernel &get_current_kernel() {
    TC_ASSERT(current_kernel);
    return *current_kernel;
  }
};

using Kernel = Program::Kernel;

TLANG_NAMESPACE_END
