#include <taichi/common/util.h>
#include <dlfcn.h>

TC_NAMESPACE_BEGIN

namespace Tlang {

template <typename T>
using Handle = std::shared_ptr<T>;

class Node {
 public:
  enum class Type : int { mul, add, sub, div, load, store, combine };

  std::vector<Handle<Node>> ch;  // Four child max
  Type type;
  std::string var_name;
  int stream_id, stride, offset;

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

  Expr(Handle<Node> node) : node(node) {
  }

#define BINARY_OP(op, name)                                            \
  Expr operator op(const Expr &o) {                                    \
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

Expr load(int stream_id, int stride, int offset) {
  auto n = std::make_shared<Node>(NodeType::load);
  n->stream_id = stream_id;
  n->stride = stride;
  n->offset = offset;
  return Expr(n);
}

class CodeGen {
  int var_count;
  std::string code;

 public:
  CodeGen() : var_count(0) {
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
    code +=
        "extern \"C\" void func(float32 *stream00, float32 *stream01, float32 "
        "*stream02, "
        "int n) {\n";
    code += "for (int i = 0; i < n; i += 8) {\n";
    visit(e.node);
    code += "}\n}\n";
    return code;
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
    if (node->type == NodeType::add) {
      code += fmt::format("auto {} = {} + {};\n", node->var_name,
                          node->ch[0]->var_name, node->ch[1]->var_name);
    } else if (node->type == NodeType::mul) {
      code += fmt::format("auto {} = {} * {};\n", node->var_name,
                          node->ch[0]->var_name, node->ch[1]->var_name);
    } else if (node->type == NodeType::sub) {
      code += fmt::format("auto {} = {} - {};\n", node->var_name,
                          node->ch[0]->var_name, node->ch[1]->var_name);
    } else if (node->type == NodeType::div) {
      code += fmt::format("auto {} = {} / {};\n", node->var_name,
                          node->ch[0]->var_name, node->ch[1]->var_name);
    } else if (node->type == NodeType::load) {
      auto stream_name = fmt::format("stream{:02d}", node->stream_id);
      code +=
          fmt::format("__m256 {} = _mm256_load_ps(&{}[{} * i + {}]);\n",
                      node->var_name, stream_name, node->stride, node->offset);
    } else if (node->type == NodeType::store) {
      auto stream_name = fmt::format("stream{:02d}", node->stream_id);
      code +=
          fmt::format("_mm256_store_ps(&{}[{} * i + {}], {});\n", stream_name,
                      node->stride, node->offset, node->ch[0]->var_name);
    } else if (node->type == NodeType::combine) {
      // do nothing
    } else {
      TC_NOT_IMPLEMENTED;
    }
  }

  FunctionType get(const Expr &e) {
    run(e);
    {
      std::ofstream of("tmp.cpp");
      of << code;
    }
    std::system("clang-format-4.0 -i tmp.cpp");
    auto compile_ret =
        std::system("g++ tmp.cpp -std=c++14 -shared -fPIC -o tmp.so -march=native");
    TC_ASSERT(compile_ret == 0);
    auto dll = dlopen("./tmp.so", RTLD_LAZY);
    TC_ASSERT(dll != nullptr);
    auto ret = dlsym(dll, "func");
    TC_ASSERT(ret != nullptr);
    return (FunctionType)ret;
  }
};
}

TC_NAMESPACE_END
