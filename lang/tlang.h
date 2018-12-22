#pragma once

#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>
#include <dlfcn.h>

#include "visitor.h"
#include "expr.h"
#include "address.h"
#include "memory_allocator.h"
#include "vectorizer.h"
#include "math.h"
#include "../headers/common.h"

TC_NAMESPACE_BEGIN

namespace Tlang {

class CodeGenBase : public Visitor {
 public:
  int var_count;
  std::string code, code_suffix;
  std::map<NodeType, std::string> binary_ops;
  std::string folder;
  std::string func_name;
  int num_groups;
  int id;
  std::string suffix;
  using FunctionType = void (*)(Context);

  static int get_code_gen_id() {
    static int id = 0;
    TC_ASSERT(id < 10000);
    return id++;
  }

  CodeGenBase() : Visitor(Visitor::Order::child_first) {
    code = "";
    id = get_code_gen_id();
    func_name = fmt::format("func{:06d}", id);
    binary_ops[NodeType::add] = "+";
    binary_ops[NodeType::sub] = "-";
    binary_ops[NodeType::mul] = "*";
    binary_ops[NodeType::div] = "/";
    binary_ops[NodeType::mod] = "%";
    folder = "_tlang_cache/";
    create_directories(folder);
    var_count = 0;
    code_suffix = "\n";
  }

  std::string create_variable() {
    TC_ASSERT(var_count < 10000);
    return fmt::format("var_{:04d}", var_count++);
  }

  std::string get_scalar_suffix(int i) {
    return fmt::format("_{:03d}", i);
  }

  std::string get_source_fn() {
    return fmt::format("{}/tmp{:04d}.{}", folder, id, suffix);
  }

  std::string get_project_fn() {
    return fmt::format("{}/projects/taichi_lang/", get_repo_dir());
  }

  std::string get_library_fn() {
#if defined(TC_PLATFORM_OSX)
    // Note: use .so here will lead to wired behavior...
    return fmt::format("{}/tmp{:04d}.dylib", folder, id);
#else
    return fmt::format("{}/tmp{:04d}.so", folder, id);
#endif
  }

  template <typename... Args>
  void emit_code(std::string f, Args &&... args) {
    if (sizeof...(args)) {
      code += fmt::format(f, std::forward<Args>(args)...) + code_suffix;
    } else {
      code += f + code_suffix;
    }
  }

  void write_code_to_file() {
    {
      std::ofstream of(get_source_fn());
      of << code;
    }
    trash(std::system(
        fmt::format("cp {} {}_unformated", get_source_fn(), get_source_fn())
            .c_str()));
    auto format_ret =
        std::system(fmt::format("clang-format -i {}", get_source_fn()).c_str());
    trash(format_ret);
  }

  FunctionType load_function() {
    auto dll = dlopen(("./" + get_library_fn()).c_str(), RTLD_LAZY);
    TC_ASSERT(dll != nullptr);
    auto ret = dlsym(dll, func_name.c_str());
    TC_ASSERT(ret != nullptr);
    return (FunctionType)ret;
  }
};

struct CompileConfig {
  enum class Arch { x86_64, gpu };

  Arch arch;
  int simd_width;
  int group_size;

  CompileConfig() {
    arch = Arch::x86_64;
    simd_width = -1;
    group_size = -1;
  }
};

enum class Device { cpu, gpu };

class AlignedAllocator {
  std::vector<uint8> _data;
  void *data;
  void *_cuda_data;

 public:
  Device device;

  AlignedAllocator() {
    data = nullptr;
  }

  AlignedAllocator(std::size_t size, Device device = Device::cpu);

  ~AlignedAllocator();

  bool initialized() const {
    return data != nullptr;
  }

  template <typename T = void>
  T *get() {
    TC_ASSERT(initialized());
    return reinterpret_cast<T *>(data);
  }

  AlignedAllocator operator=(const AlignedAllocator &) = delete;

  AlignedAllocator(AlignedAllocator &&o) noexcept {
    (*this) = std::move(o);
  }

  AlignedAllocator &operator=(AlignedAllocator &&o) noexcept {
    std::swap(_data, o._data);
    data = o.data;
    o.data = nullptr;
    device = o.device;
    return *this;
  }
};

using Arch = CompileConfig::Arch;

inline int default_simd_width(Arch arch) {
  if (arch == CompileConfig::Arch::x86_64) {
    return 8;  // AVX2
  } else if (arch == CompileConfig::Arch::gpu) {
    return 32;
  } else {
    TC_NOT_IMPLEMENTED;
    return -1;
  }
}

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
  CodeGenBase::FunctionType function;
  int n;
  MemoryAllocator alloc;
  Device device;
  std::vector<AlignedAllocator> buffers;
  Expr ret;
  std::vector<Cache> caches;

  Program(CompileConfig::Arch arch, int n) : n(n) {
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


real get_cpu_frequency();

extern real default_measurement_time;

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second = default_measurement_time);

inline std::pair<int64, int64> range(int64 start, int64 end) {
  return {start, end};
}

using ForBody = std::function<void()>;
inline void for_loop(Index &index,
                     std::pair<int64, int64> r,
                     const ForBody &body) {
  auto &prog = get_current_program();
  TC_ASSERT(r.first == 0);
  TC_ASSERT(prog.n == r.second);
  body();
}
}  // namespace Tlang

TC_NAMESPACE_END

/*
 Expr should be what the users play with.
   Simply a ref-counted pointer to nodes, with some operator overloading for
 users to program Node is the IR node, with computational graph connectivity,
 imm, op type etc.

 No double support this time.
 */
