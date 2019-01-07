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
  int id;
  DataType dt;

  std::vector<Expr> store_exprs;

  Adapter(int id) : id(id) {
    input_group_size = -1;
    stores = Expr::create(NodeType::combine);
  }

  bool initialized() {
    return input_group_size != -1;
  }

  Adapter &convert(Expr &e) {
    TC_ASSERT(initialized());
    TC_ASSERT(e->type != NodeType::pointer);
    if (counter == 0) {
      dt = e->data_type;
    } else {
      TC_ASSERT_INFO(dt == e->data_type,
                     "An adapter can have only one data type");
    }
    int i = counter++;
    auto n = Expr::create(NodeType::adapter_store, e, Expr::create_imm(id),
                          Expr::create_imm(i / input_group_size));
    stores->ch.push_back(n);
    e.set(Expr::create(NodeType::adapter_load, Expr::create_imm(id),
                       Expr::create_imm(i)));
    e->data_type = dt;
    return *this;
  }

  template <typename... Args>
  Adapter &convert(Expr &e, Args &&... args) {
    convert(e);
    convert(std::forward<Args>(args)...);
    return *this;
  }

  Adapter &set(int input_group_size) {
    this->input_group_size = input_group_size;
    return *this;
  }
};

struct Program {
  // Should be copiable
  struct Kernel {
    Program &program;
    FunctionType compiled;
    std::vector<Adapter> adapters;
    Expr ret;
    int parallel_instances;
    int simd_lanes;
    int output_group_size;
    bool has_touch;

    Kernel(Program &program, std::function<void()> func) : program(program) {
      has_touch = false;
      parallel_instances = -1;
      simd_lanes = -1;
      output_group_size = -1;
      program.start_function_definition(this);
      ret = Expr::create(NodeType::combine);
      func();

      if (output_group_size == -1) {
        output_group_size = 1;
      }
      if (parallel_instances == -1) {
        int minimal_group_size = output_group_size;
        for (auto &ad : adapters) {
          minimal_group_size =
              std::min(minimal_group_size, ad.input_group_size);
        }
        TC_ASSERT(
            default_simd_width(program.config.arch) % minimal_group_size == 0);
        parallel_instances =
            default_simd_width(program.config.arch) / minimal_group_size;
      }
      if (simd_lanes == -1) {
        simd_lanes = output_group_size * parallel_instances;
      }

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

    Adapter &adapter(int i) {
      while ((int)adapters.size() <= i) {
        adapters.push_back(Adapter((int)adapters.size()));
      }
      return adapters[i];
    }
  };

  Kernel *current_kernel;
  SNode *current_snode;
  SNode *snode_root;
  void *data_structure;
  CompileConfig config;
  Device device;

  std::vector<AlignedAllocator> buffers;
  std::vector<Kernel> functions;
  int index_counter;

  std::string layout_fn;

  Context get_context() {
    Context context;
    context.buffers[0] = data_structure;
    return context;
  }

  Program(Arch arch = Arch::x86_64) {
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
    config.simd_width = default_simd_width(arch);
    current_kernel = nullptr;
    snode_root = nullptr;
    index_counter = 0;
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
    Expr::set_allow_store(false);
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

  Expr store(const Expr &ad, const Expr &e) {
    return get_current_kernel().ret.store(ad, e);
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
