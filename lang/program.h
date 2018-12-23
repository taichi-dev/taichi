#pragma once

#include "util.h"
#include "memory_allocator.h"

namespace taichi::Tlang {

struct Program;
extern Program *current_program;

TC_FORCE_INLINE Program &get_current_program() {
  return *current_program;
}

struct Adapter {
  Expr stores;
  int counter = 0;
  int input_group_size;
  int output_group_size;

  Adapter() {
    input_group_size = -1;
    output_group_size = -1;
    stores = Expr::create(Expr::Type::combine);
  }

  void convert(Expr &e) {
    int i = counter++;
    auto n = Expr::create(NodeType::adapter_store, e, Expr::create_imm(i));
    stores->ch.push_back(n);
    e = Expr::create(NodeType::adapter_load, Expr::create_imm(i));
  }

  void set(int input_group_size, int output_group_size) {
    this->input_group_size = input_group_size;
    this->output_group_size = output_group_size;
  }
};

struct Program {
  int n;

  CompileConfig config;
  FunctionType function;
  MemoryAllocator alloc;
  Device device;
  Expr ret;

  std::vector<AlignedAllocator> buffers;
  std::vector<Adapter> adapters;

  Program(Arch arch, int n) : n(n) {
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
    function = nullptr;
  }

  ~Program() {
    current_program = nullptr;
  }

  Adapter &adapter(int i) {
    while ((int)adapters.size() <= i) {
      adapters.push_back(Adapter());
    }
    return adapters[i];
  }

  Expr store(const Expr &ad, const Expr &e) {
    return ret.store(ad, e);
  }

  AddrNode &buffer(int i) {
    return alloc.buffer(i);
  }

  void set_n(int n) {
    this->n = n;
    // TODO: resize buffers
    TC_NOT_IMPLEMENTED
  }

  int num_buffers() {
    return (int)alloc.root->ch.size();
  }

  void operator()() {
    if (function == nullptr) {
      compile();
    }
    Context context;
    for (int i = 0; i < num_buffers(); i++) {
      allocate_buffer(i);
      context.buffers[i] = buffers[i].get();
    }
    context.ranges[0] = n;
    function(context);
  }

  void operator()(Context context) {
    if (function == nullptr) {
      compile();
    }
    function(context);
  }

  void compile();

  void allocate_buffer(int i) {
    while ((int)buffers.size() <= i) {
      buffers.emplace_back();
    }
    if (!buffers[i].initialized()) {
      buffers[i] = std::move(AlignedAllocator(
          alloc.buffer(i).num_variables * n * sizeof(float32), device));
    }
  }

  float32 &data(Expr &expr, int i) {
    auto &addr = expr->get_address_();  // TODO:...
    TC_ASSERT(addr.initialized());
    allocate_buffer(addr.buffer_id);
    return buffers[addr.buffer_id].get<float32>()[addr.eval(i, n)];
  }

  void materialize_layout() {
    alloc.materialize();
  }

  void swap_buffers(int i, int j) {
    std::swap(buffers[i], buffers[j]);
  }
};
}
