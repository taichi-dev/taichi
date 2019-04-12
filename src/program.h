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
    std::unique_ptr<IRNode> ir_holder;
    IRNode *ir;
    Program &program;
    FunctionType compiled;
    std::string name;

    Kernel(Program &program, std::function<void()> func) : program(program) {
      compiled = nullptr;
      benchmarking = false;
      context = std::make_unique<FrontendContext>();
      ir_holder = context->get_root();
      ir = ir_holder.get();

      program.current_kernel = this;
      program.start_function_definition(this);
      func();
      program.end_function_definition();
      program.current_kernel = nullptr;
    }

    void compile() {
      program.current_kernel = this;
      compiled = program.compile(*this);
      program.current_kernel = nullptr;
    }

    void operator()() {
      if (!compiled)
        compile();
      auto c = program.get_context();
      compiled(c);
    }

    bool benchmarking;
  };

  Kernel *current_kernel;
  SNode *current_snode;
  SNode *snode_root;
  void *data_structure;
  CompileConfig config;

  std::vector<std::unique_ptr<Kernel>> functions;
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

  Kernel &kernel(const std::function<void()> &body) {
    // Expr::set_allow_store(true);
    auto func = std::make_unique<Kernel>(*this, body);
    // Expr::set_allow_store(false);
    functions.emplace_back(std::move(func));
    current_snode = nullptr;
    return *functions.back();
  }

  void start_function_definition(Kernel *func) {
    current_kernel = func;
  }

  void end_function_definition() {
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
