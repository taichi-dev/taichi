#pragma once

#include "util.h"
#include "structural_node.h"

TLANG_NAMESPACE_BEGIN

struct Program;
extern Program *current_program;
extern SNode root;

TC_FORCE_INLINE Program &get_current_program() {
  return *current_program;
}

struct Adapter {
  Expr stores;
  int counter = 0;
  int input_group_size;
  int output_group_size;
  int id;
  DataType dt;

  std::vector<Expr> store_exprs;

  Adapter(int id) : id(id) {
    input_group_size = -1;
    output_group_size = -1;
    stores = Expr::create(NodeType::combine);
  }

  bool initialized() {
    return input_group_size != -1 && output_group_size != -1;
  }

  Adapter &convert(Expr &e) {
    TC_ASSERT(initialized());
    TC_ASSERT(e->type != NodeType::pointer);
    if (counter == 0) {
      dt = e->data_type;
    } else {
      TC_ASSERT(dt == e->data_type);
    }
    int i = counter++;
    auto n = Expr::create(NodeType::adapter_store, e, Expr::create_imm(id),
                          Expr::create_imm(i));
    stores->ch.push_back(n);
    e.set(Expr::create(NodeType::adapter_load, Expr::create_imm(id),
                       Expr::create_imm(i)));
    e->data_type = dt;
    TC_P(counter);
    TC_P(input_group_size);
    store_exprs.resize(counter / input_group_size);
    return *this;
  }

  template <typename... Args>
  Adapter &convert(Expr &e, Args &&... args) {
    convert(e);
    convert(std::forward<Args>(args)...);
    return *this;
  }

  Adapter &set(int input_group_size, int output_group_size = -1) {
    this->input_group_size = input_group_size;
    this->output_group_size = output_group_size;
    return *this;
  }
};

struct Program {
  // Should be copiable
  struct Kernel {
    Program &program;
    FunctionType compiled;
    Expr ret;
    int stride;

    Kernel(Program &program, std::function<void()> func) : program(program) {
      stride = 1;
      program.start_function_definition(this);
      ret = Expr(nullptr);
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
  Device device;

  std::vector<AlignedAllocator> buffers;
  std::vector<Adapter> adapters;
  std::vector<Kernel> functions;

  std::string layout_fn;

  Context get_context() {
    Context context;
    context.buffers[0] = data_structure;
    return context;
  }

  Program(Arch arch) {
    Node::reset_counter();
    TC_ASSERT(current_program == nullptr);
    current_program = this;
    config.arch = arch;
    if (config.arch == Arch::x86_64) {
      device = Device::cpu;
    } else if (config.arch == Arch::gpu) {
      device = Device::gpu;
    } else {
      TC_NOT_IMPLEMENTED;
    }
    current_kernel = nullptr;
    snode_root = nullptr;
  }

  ~Program() {
    current_program = nullptr;
  }

  void layout(std::function<void()> func) {
    root = SNode(0, SNodeType::forked);
    snode_root = &root;
    func();
    materialize_layout();
  }

  Kernel def(const std::function<void()> &body) {
    Expr::set_allow_store(true);
    auto func = Kernel(*this, body);
    functions.push_back(func);
    Expr::set_allow_store(false);
    return func;
  }

  Kernel kernel(Expr exp, const std::function<void()> &body) {
    return kernel(exp->new_addresses(0), body);
  }

  Kernel kernel(SNode *snode, const std::function<void()> &body) {
    Expr::set_allow_store(true);
    current_snode = snode;
    auto func = Kernel(*this, body);
    functions.push_back(func);
    Expr::set_allow_store(false);
    current_snode = nullptr;
    return func;
  }

  void start_function_definition(Kernel *func) {
    current_kernel = func;
  }

  void end_function_definition() {
    current_kernel = nullptr;
  }

  Adapter &adapter(int i) {
    while ((int)adapters.size() <= i) {
      adapters.push_back(Adapter((int)adapters.size()));
    }
    return adapters[i];
  }

  Expr store(const Expr &ad, const Expr &e) {
    return get_current_kernel().ret.store(ad, e);
  }

  FunctionType compile(Kernel &kernel);

  template <typename T = float32>
  T &data(Expr &expr, int i) {
    if (get_data_type<T>() != expr->data_type) {
      TC_ERROR("Cannot access type {} as type {}",
               data_type_name(expr->data_type),
               data_type_name(get_data_type<T>()));
    }
    return expr.val<T>(i);
  }

  void materialize_layout();

  inline Kernel &get_current_kernel() {
    TC_ASSERT(current_kernel);
    return *current_kernel;
  }

};

using Kernel = Program::Kernel;

TLANG_NAMESPACE_END
