#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <dlfcn.h>

TC_NAMESPACE_BEGIN

namespace Tlang {

template <typename T>
using Handle = std::shared_ptr<T>;

class Node {
 public:
  enum class Type : int { mul, add, sub, div, load, store, combine, constant };

  std::vector<Handle<Node>> ch;  // Four child max
  Type type;
  std::string var_name;
  int stream_id, stride, offset;
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

  void store(const Expr &e, int stream_id, int stride, int offset) {
    if (!node) {
      node = std::make_shared<Node>(NodeType::combine);
    }
    auto n = std::make_shared<Node>(NodeType::store);
    n->ch.push_back(e.node);
    n->stream_id = stream_id;
    n->stride = stride;
    n->offset = offset;
    Expr store_e(n);
    node->ch.push_back(n);
  }
};

inline Expr load(int stream_id, int stride, int offset) {
  auto n = std::make_shared<Node>(NodeType::load);
  n->stream_id = stream_id;
  n->stride = stride;
  n->offset = offset;
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
    code = "#include <immintrin.h>\n\n";
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
      auto stream_name = fmt::format("stream{:02d}", node->stream_id);
      if (mode == Mode::vector) {
        std::string load_instr =
            simd_width == 8 ? "_mm256_load_ps" : "_mm512_load_ps";
        code +=
            fmt::format("auto {} = {}(&{}[{} * i + {}]);\n", node->var_name,
                        load_instr, stream_name, node->stride, node->offset);
      } else {
        for (int i = 0; i < simd_width; i++) {
          auto suf = get_scalar_suffix(i);
          code += fmt::format("auto {} = {}[{} * i + {} + {}];\n",
                              node->var_name + suf, stream_name, node->stride,
                              node->offset, i);
        }
      }
    } else if (node->type == NodeType::store) {
      auto stream_name = fmt::format("stream{:02d}", node->stream_id);
      if (mode == Mode::vector) {
        std::string store_instr =
            simd_width == 8 ? "_mm256_store_ps" : "_mm512_store_ps";
        code +=
            fmt::format("{}(&{}[{} * i + {}], {});\n", store_instr, stream_name,
                        node->stride, node->offset, node->ch[0]->var_name);
      } else {
        for (int i = 0; i < simd_width; i++) {
          auto suf = get_scalar_suffix(i);
          code += fmt::format("{}[{} * i + {} + {}] = {};\n", stream_name,
                              node->stride, node->offset, i,
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
    return fmt::format("{}/tmp{:04d}.so", folder, id);
  }

  FunctionType get(const Expr &e) {
    run(e);
    {
      std::ofstream of(get_source_fn());
      of << code;
    }
    std::system(fmt::format("clang-format -i {}", get_source_fn()).c_str());
    auto cmd =
        fmt::format("g++ {} -std=c++14 -shared -fPIC -O3 -march=native -o {}",
                    get_source_fn(), get_library_fn());
    auto compile_ret = std::system(cmd.c_str());
    TC_ASSERT(compile_ret == 0);
    system(
        fmt::format("objdump {} -d > {}.s", get_library_fn(), get_library_fn())
            .c_str());
    auto dll = dlopen(("./" + get_library_fn()).c_str(), RTLD_LAZY);
    TC_ASSERT(dll != nullptr);
    auto ret = dlsym(dll, func_name.c_str());
    TC_ASSERT(ret != nullptr);
    return (FunctionType)ret;
  }
};
}  // namespace Tlang

TC_NAMESPACE_END
