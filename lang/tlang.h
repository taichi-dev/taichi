#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <dlfcn.h>

TC_NAMESPACE_BEGIN

namespace Tlang {

template <typename T>
using Handle = std::shared_ptr<T>;

struct Address {
  int64 stream_id;
  int64 coeff_i;
  int64 coeff_imax;
  int64 coeff_const;  // offset

  // AOSOA: i / a * b
  int64 coeff_aosoa_group_size;
  int64 coeff_aosoa_stride;

  Address() {
    stream_id = -1;
    coeff_i = 0;
    coeff_imax = 0;
    coeff_const = 0;
    coeff_aosoa_group_size = 0;
    coeff_aosoa_stride = 0;
  }

  bool initialized() {
    return stream_id != -1;
  }

  TC_FORCE_INLINE bool same_type(Address o) {
    return stream_id == o.stream_id && coeff_i == o.coeff_i &&
           coeff_imax == o.coeff_imax &&
           coeff_aosoa_group_size == o.coeff_aosoa_group_size &&
           coeff_aosoa_stride == o.coeff_aosoa_group_size;
  }

  TC_FORCE_INLINE bool operator==(Address o) {
    return stream_id == o.stream_id && coeff_i == o.coeff_i &&
           coeff_imax == o.coeff_imax && coeff_const == o.coeff_const &&
           coeff_aosoa_group_size == o.coeff_aosoa_group_size &&
           coeff_aosoa_stride == o.coeff_aosoa_group_size;
  }

  TC_FORCE_INLINE int64 offset() {
    return coeff_const;
  }
};

class Node {
 public:
  enum class Type : int { mul, add, sub, div, load, store, combine, constant };

  Address addr;
  std::vector<Handle<Node>> ch;  // Four child max
  Type type;
  std::string var_name;
  float64 value;

  Node(Type type) : type(type) {
  }

  Node(Type type, Handle<Node> ch0, Handle<Node> ch1) : type(type) {
    ch.resize(2);
    ch[0] = ch0;
    ch[1] = ch1;
  }
};

using NodeType = Node::Type;

// Reference counted...
class Expr {
 public:
  Handle<Node> node;

  Expr() {
  }

  Expr(float64 val) {
    // cretea a constant node
    node = std::make_shared<Node>(NodeType::constant);
    node->value = val;
  }

  Expr(Handle<Node> node) : node(node) {
  }

#define BINARY_OP(op, name)                                            \
  Expr operator op(const Expr &o) const {                              \
    return Expr(std::make_shared<Node>(NodeType::name, node, o.node)); \
  }

  BINARY_OP(*, mul);
  BINARY_OP(+, add);
  BINARY_OP(-, sub);
  BINARY_OP(/, div);
#undef BINARY

  void store(const Expr &e, Address addr) {
    if (!node) {
      node = std::make_shared<Node>(NodeType::combine);
    }
    auto n = std::make_shared<Node>(NodeType::store);
    n->ch.push_back(e.node);
    n->addr = addr;
    Expr store_e(n);
    node->ch.push_back(n);
  }
};

inline Expr load(Address addr) {
  auto n = std::make_shared<Node>(NodeType::load);
  TC_ASSERT(addr.initialized());
  n->addr = addr;
  TC_ASSERT(0 <= addr.stream_id && addr.stream_id < 3);
  return Expr(n);
}

inline int get_code_gen_id() {
  static int id = 0;
  TC_ASSERT(id < 10000);
  return id++;
}

class CodeGen {
  int var_count;
  std::string code;

 public:
  std::string func_name;
  enum class Mode : int { scalar, vector };

  Mode mode;
  int simd_width;
  int id;
  std::map<NodeType, std::string> binary_ops;
  std::string folder;

  CodeGen(Mode mode = Mode::vector, int simd_width = 8)
      : var_count(0), mode(mode), simd_width(simd_width) {
    id = get_code_gen_id();
    func_name = fmt::format("func{:06d}", id);
    binary_ops[NodeType::add] = "+";
    binary_ops[NodeType::sub] = "-";
    binary_ops[NodeType::mul] = "*";
    binary_ops[NodeType::div] = "/";
    folder = "_tlang_cache/";
    create_directories(folder);
  }

  std::string create_variable() {
    TC_ASSERT(var_count < 10000);
    return fmt::format("var_{:04d}", var_count++);
  }

  using FunctionType = void (*)(float32 *, float32 *, float32 *, int);

  std::string run(const Expr &e) {
    code = "#include <immintrin.h>\n#include <cstdio>\n";
    code += "using float32 = float;\n";
    code += "using float64 = double;\n\n";
    code += "extern \"C\" void " + func_name +
            "(float32 *stream00, float32 *stream01, float32 "
            "*stream02, "
            "int n) {\n";
    code += fmt::format("for (int i = 0; i < n; i += {}) {{\n", simd_width);
    visit(e.node);
    code += "}\n}\n";
    return code;
  }

  std::string get_scalar_suffix(int i) {
    return fmt::format("_{:03d}", i);
  }

  void visit(const Handle<Node> &node) {
    for (auto &c : node->ch) {
      if (c)
        visit(c);
    }
    if (node->var_name == "")
      node->var_name = create_variable();
    else
      return;  // visited
    if (binary_ops.find(node->type) != binary_ops.end()) {
      auto op = binary_ops[node->type];
      if (mode == Mode::vector) {
        code += fmt::format("auto {} = {} {} {};\n", node->var_name,
                            node->ch[0]->var_name, op, node->ch[1]->var_name);
      } else if (mode == Mode::scalar) {
        for (int i = 0; i < simd_width; i++) {
          auto suf = get_scalar_suffix(i);
          code += fmt::format("auto {} = {} {} {};\n", node->var_name + suf,
                              node->ch[0]->var_name + suf, op,
                              node->ch[1]->var_name + suf);
        }
      }
    } else if (node->type == NodeType::load) {
      auto stream_name = fmt::format("stream{:02d}", node->addr.stream_id);
      if (mode == Mode::vector) {
        std::string load_instr =
            simd_width == 8 ? "_mm256_load_ps" : "_mm512_load_ps";
        code +=
            fmt::format("auto {} = {}(&{}[{} * i + {}]);\n", node->var_name,
                        load_instr, stream_name, node->addr.coeff_i, node->addr.coeff_const);
      } else {
        for (int i = 0; i < simd_width; i++) {
          auto suf = get_scalar_suffix(i);
          code += fmt::format("auto {} = {}[{} * i + {} + {}];\n",
                              node->var_name + suf, stream_name, node->addr.coeff_i,
                              node->addr.coeff_const, i);
        }
      }
    } else if (node->type == NodeType::store) {
      auto stream_name = fmt::format("stream{:02d}", node->addr.stream_id);
      if (mode == Mode::vector) {
        std::string store_instr =
            simd_width == 8 ? "_mm256_store_ps" : "_mm512_store_ps";
        code +=
            fmt::format("{}(&{}[{} * i + {}], {});\n", store_instr, stream_name,
                        node->addr.coeff_i, node->addr.coeff_const, node->ch[0]->var_name);
      } else {
        for (int i = 0; i < simd_width; i++) {
          auto suf = get_scalar_suffix(i);
          code += fmt::format("{}[{} * i + {} + {}] = {};\n", stream_name,
                              node->addr.coeff_i, node->addr.coeff_const, i,
                              node->ch[0]->var_name + suf);
        }
      }
    } else if (node->type == NodeType::combine) {
      // do nothing
    } else {
      TC_NOT_IMPLEMENTED;
    }
  }

  std::string get_source_fn() {
    return fmt::format("{}/tmp{:04d}.cpp", folder, id);
  }

  std::string get_library_fn() {
#if defined(TC_PLATFORM_OSX)
    // Note: use .so here will lead to wired behavior...
    return fmt::format("{}/tmp{:04d}.dylib", folder, id);
#else
    return fmt::format("{}/tmp{:04d}.so", folder, id);
#endif
  }

  FunctionType get(const Expr &e, int group_size = 4) {
    SLP(e, group_size);
    run(e);
    {
      std::ofstream of(get_source_fn());
      of << code;
    }
    std::system(fmt::format("clang-format -i {}", get_source_fn()).c_str());
    auto cmd =
        fmt::format("g++ {} -std=c++14 -shared -fPIC -O3 -march=native -D_GLIBCXX_USE_CXX11_ABI=0 -o {}",
                    get_source_fn(), get_library_fn());
    auto compile_ret = std::system(cmd.c_str());
    TC_ASSERT(compile_ret == 0);
    /*
    system(
        fmt::format("objdump {} -d > {}.s", get_library_fn(), get_library_fn())
            .c_str());
            */
    auto dll = dlopen(("./" + get_library_fn()).c_str(), RTLD_LAZY);
    TC_ASSERT(dll != nullptr);
    auto ret = dlsym(dll, func_name.c_str());
    TC_ASSERT(ret != nullptr);
    return (FunctionType)ret;
  }

  bool prior_to(Expr &a, Expr &b) {
    auto address1 = a.node->addr;
    auto address2 = b.node->addr;
    return address1.same_type(address2) &&
           address1.offset() + 1 == address2.offset();
  }

  std::vector<Expr> extract_instructions(Expr root_expr) {
    std::vector<Expr> inst;

    std::function<void(Expr)> walk = [&](Expr expr) -> void {
      inst.push_back(expr);
      for (auto &ch : expr.node->ch) {
        walk(ch);
      }
    };

    walk(root_expr);

    return inst;
  }

  void propagate() {
  }

  std::vector<Expr> inst;
  std::vector<std::vector<int>> groups;
  std::vector<bool> grouped;

  std::vector<int> continuous_loads(int i) {
    std::vector<int> ret;
    if (grouped[i] || inst[i].node->type != NodeType::load) {
      return ret;
    }
    ret.push_back(i);
    while (1) {
      bool found = false;
      for (int j = 0; j < inst.size(); j++) {
        if (grouped[j] || i == j || inst[i].node->type != NodeType::load) {
          continue;
        }
        if (prior_to(inst[i], inst[j])) {
          ret.push_back(j);
          i = j;
          found = true;
          break;
        }
      }
      if (!found) {
        break;
      }
    }
    return ret;
  }

  void SLP(Expr expr, int group_size) {
    return;
    inst = extract_instructions(expr);
    TC_INFO("# instructions = {}", inst.size());
    grouped = std::vector<bool>(inst.size(), false);

    while (true) {
      // Enumerate the instructions
      int maximum_length = 0;
      int inst_with_max_length = -1;
      std::vector<int> C;
      for (int i = 0; i < inst.size(); i++) {
        auto c = continuous_loads(i);
        if (c.size() > maximum_length) {
          maximum_length = c.size();
          inst_with_max_length = i;
          C = c;
        }
      }
      TC_P(C);

      // Extend
      if (inst_with_max_length != -1) {
        // Pack
        TC_WARN_IF(C.size() % group_size != 0, "C.size() = {}", C.size());
        groups.push_back(std::vector<int>());
        for (int i = 0; i < C.size(); i++) {
          grouped[C[i]] = true;
          groups.back().push_back(C[i]);
        }
      } else {
        break;
      }
    }

    TC_INFO("# groups {}", groups.size());
    for (int i = 0; i < groups.size(); i++) {
      TC_INFO("Group {} size = {}", i, groups[i].size());
    }

    /*
    while (true) {
      propagate();
    }
    */
  }
};
}  // namespace Tlang

TC_NAMESPACE_END
