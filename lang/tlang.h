#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>
#include <dlfcn.h>

#include "visitor.h"
#include "expr.h"
#include "address.h"
#include "memory_allocator.h"
#include "../headers/common.h"

TC_NAMESPACE_BEGIN

namespace Tlang {

inline int get_code_gen_id() {
  static int id = 0;
  TC_ASSERT(id < 10000);
  return id++;
}

class Vectorizer : public Visitor {
 public:
  std::map<Expr, Expr> scalar_to_vector;
  int simd_width;
  int group_size;
  int num_groups;

  Vectorizer(int simd_width)
      : Visitor(Visitor::Order::parent_first), simd_width(simd_width) {
  }

  void sort(Expr &expr) {
    auto ch = expr->ch;
    std::vector<Expr> sorted;

    while (!ch.empty()) {
      std::vector<Expr> group;
      group.push_back(ch[0]);
      ch.erase(ch.begin());
      bool progress = false;
      while (true) { // grow
        bool found = false;
        for (int i = 0; i < ch.size(); i++) { // search
          if (prior_to(ch[i]->addr(), group.front()->addr())) {
            group.insert(group.begin(), ch[i]);
            ch.erase(ch.begin() + i);
            found = true;
            break;
          }
          if (prior_to(group.back()->addr(), ch[i]->addr())) {
            group.insert(group.end(), ch[i]);
            ch.erase(ch.begin() + i);
            found = true;
            break;
          }
        }
        if (!found) {
          break;
        }
      }
      TC_ASSERT(group.size() % group_size == 0);
      sorted.insert(sorted.end(), group.begin(), group.end());
    }
    expr->ch = sorted;
  }

  Expr run(Expr &expr, int group_size) {
    this->group_size = group_size;
    this->num_groups = simd_width / group_size;
    TC_ASSERT(group_size * num_groups == simd_width);

    sort(expr);

    scalar_to_vector.clear();
    // expr should be a ret Op, with its children store Ops.
    // The stores are repeated by a factor of 'pack_size'
    TC_ASSERT(expr->ch.size() % group_size == 0);
    TC_ASSERT(expr->type == NodeType::combine);
    // Create the root group
    auto combined = Expr::create(NodeType::combine);
    combined->is_vectorized = true;
    // for each batch (group)
    for (int k = 0; k < (int)expr->ch.size() / group_size; k++) {
      auto root = Expr::create(NodeType::store);
      root->is_vectorized = true;
      bool has_prior_to = false, has_same = false;
      for (int i = 0; i < group_size; i++) {
        auto ch = expr->ch[k * group_size + i];
        TC_ASSERT(ch->type == NodeType::store);
        root->members.push_back(ch);  // put scalar inst into vector members
        TC_ASSERT(i < (int)root->members.size());
        if (i > k * group_size) {
          if (prior_to(root->members[i - 1]->ch[0], root->members[i]->ch[0])) {
            has_prior_to = true;
          } else if (root->members[i - 1]->ch[0]->get_address() ==
                     root->members[i]->ch[0]->get_address()) {
            has_same = true;
          } else {
            TC_P(root->members[i - 1]->ch[0]->get_address());
            TC_P(root->members[i]->ch[0]->get_address());
            TC_ERROR(
                "Addresses in SIMD load should be either identical or "
                "neighbouring.");
          }
        }
      }
      TC_ASSERT(!(has_prior_to && has_same));
      // TC_P(root->members.size());
      root.accept(*this);
      combined->ch.push_back(root);
    }
    // TC_P(combined->ch.size());
    return combined;
  }

  void visit(Expr &expr) override {
    // Note: expr may be replaced by an existing vectorized Expr
    if (scalar_to_vector.find(expr->members[0]) != scalar_to_vector.end()) {
      auto existing = scalar_to_vector[expr->members[0]];
      TC_ASSERT(existing->members.size() == expr->members.size());
      for (int i = 0; i < (int)existing->members.size(); i++) {
        TC_ASSERT(existing->members[i] == expr->members[i]);
      }
      expr = existing;
      return;
    }

    expr->is_vectorized = true;
    bool first = true;
    NodeType type;
    std::vector<std::vector<Expr>> vectorized_children;

    // Check for isomorphism
    for (auto member : expr->members) {
      // It must not appear to an existing vectorized expr
      TC_ASSERT(scalar_to_vector.find(member) == scalar_to_vector.end());
      if (first) {
        first = false;
        type = member->type;
        vectorized_children.resize(member->ch.size());
      } else {
        TC_ASSERT(type == member->type);
        TC_ASSERT(vectorized_children.size() == member->ch.size());
      }
      for (int i = 0; i < (int)member->ch.size(); i++) {
        vectorized_children[i].push_back(member->ch[i]);
      }
    }

    expr->is_vectorized = true;
    TC_ASSERT(expr->members.size() % group_size == 0);

    for (int i = 0; i < (int)vectorized_children.size(); i++) {
      // TC_P(i);
      auto ch = Expr::create(vectorized_children[i][0]->type);
      ch->members = vectorized_children[i];
      expr->ch.push_back(ch);
    }

    if (expr->type == NodeType::addr) {
      auto addr = expr->members[0]->get_address();
      if (addr.coeff_aosoa_group_size == 0 || addr.coeff_aosoa_stride == 0) {
        addr.coeff_aosoa_group_size = num_groups;
        addr.coeff_aosoa_stride = 0;
      }
      expr->get_address() = addr;
    }
  }
};

class CodeGenBase : public Visitor {
 public:
  int var_count;
  std::string code;
  std::map<NodeType, std::string> binary_ops;
  std::string folder;
  std::string func_name;
  int num_groups;
  int id;
  std::string suffix;
  using FunctionType = void (*)(Context);

  CodeGenBase() : Visitor(Visitor::Order::child_first) {
    code = "";
    id = get_code_gen_id();
    func_name = fmt::format("func{:06d}", id);
    binary_ops[NodeType::add] = "+";
    binary_ops[NodeType::sub] = "-";
    binary_ops[NodeType::mul] = "*";
    binary_ops[NodeType::div] = "/";
    folder = "_tlang_cache/";
    create_directories(folder);
    var_count = 0;
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
      code += fmt::format(f, std::forward<Args>(args)...);
    } else {
      code += f;
    }
  }

  void write_code_to_file() {
    {
      std::ofstream of(get_source_fn());
      of << code;
    }
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

struct Program {
  CompileConfig config;
  CodeGenBase::FunctionType function;
  int n;
  MemoryAllocator alloc;
  Device device;

  std::vector<AlignedAllocator> buffers;

  Expr ret;

  Expr store(const Expr &ad, const Expr &e) {
    return ret.store(ad, e);
  }

  AddrNode &buffer(int i) {
    return alloc.buffer(i);
  }

  Program(CompileConfig::Arch arch, int n) : n(n) {
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
    auto &addr = expr->get_address();
    TC_ASSERT(addr.initialized());
    allocate_buffer(addr.buffer_id);
    return buffers[addr.buffer_id].get<float32>()[addr.eval(i, n)];
  }

  void materialize_layout() {
    alloc.materialize();
  }
};

}  // namespace Tlang

namespace Tlang {
struct Matrix {
  int n, m;
  std::vector<Expr> entries;

  Matrix() {
    n = m = 0;
  }

  bool initialized() {
    return n * m >= 1;
  }

  Matrix(int n, int m = 1) : n(n), m(m) {
    TC_ASSERT(n * m >= 1);
    entries.resize(n * m, Expr());
  }

  Expr &operator()(int i, int j) {
    TC_ASSERT(0 <= i && i < n);
    TC_ASSERT(0 <= j && j < n);
    return entries[i * m + j];
  }

  const Expr &operator()(int i, int j) const {
    TC_ASSERT(0 <= i && i < n);
    TC_ASSERT(0 <= j && j < n);
    return entries[i * m + j];
  }

  Expr &operator()(int i) {
    TC_ASSERT(0 <= i && i < n * m);
    TC_ASSERT(n == 1 || m == 1);
    return entries[i];
  }

  const Expr &operator()(int i) const {
    TC_ASSERT(0 <= i && i < n * m);
    TC_ASSERT(n == 1 || m == 1);
    return entries[i];
  }

  Matrix &operator=(const Matrix &o) {
    if (initialized()) {
      n = o.n;
      m = o.m;
      entries = o.entries;
    } else {
      TC_ASSERT(n == o.n && m == o.m);
      for (int i = 0; i < (int)entries.size(); i++) {
        entries[i] = o.entries[i];
      }
    }
    return *this;
  }
};

inline Matrix operator*(const Matrix &A, const Matrix &B) {
  TC_ASSERT(A.m == B.n);
  Matrix C(A.n, B.m);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < B.m; j++) {
      C(i, j) = load(A(i, 0)) * load(B(0, j));
      for (int k = 1; k < A.m; k++) {
        C(i, j) = C(i, j) + load(A(i, k)) * load(B(k, j));
      }
    }
  }
  return C;
}

inline Matrix operator+(const Matrix &A, const Matrix &B) {
  TC_ASSERT(A.n == B.n);
  TC_ASSERT(A.m == B.m);
  Matrix C(A.n, A.m);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < A.m; j++) {
      C(i, j) = A(i, j) + B(i, j);
    }
  }
  return C;
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
