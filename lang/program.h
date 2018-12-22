#pragma once

#include "util.h"
#include "memory_allocator.h"

namespace taichi::Tlang {

struct Program;
extern Program *current_program;

TC_FORCE_INLINE Program &get_current_program() {
  return *current_program;
}

struct Cache {
  Expr stores;

  Cache() {
    stores = Expr::create(Expr::Type::combine);
  }

  void group_size() {
  }

  void store(const Expr &e, int i) {
    auto n = Expr::create(NodeType::cache_store, e, Expr::create_imm(i));
    stores->ch.push_back(n);
  }

  Expr load(int i) {
    return Expr::create(NodeType::cache_load, Expr::create_imm(i));
  }
};

struct Program {
  CompileConfig config;
  FunctionType function;
  int n;
  MemoryAllocator alloc;
  Device device;
  std::vector<AlignedAllocator> buffers;
  Expr ret;
  std::vector<Cache> caches;

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

  Cache &cache(int i) {
    while ((int)caches.size() <= i) {
      caches.push_back(Cache());
    }
    return caches[i];
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
