#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

namespace T {

template <typename T>
using Handle = std::shared_ptr<T>;

class Node {
 public:
  enum class Type : int { input, mul, add, sub, div };

  Handle<Node> ch[4];  // Four child max
  Type type;
  std::string var_name;

  Node(Type type) : type(type) {
  }

  Node(Type type, Handle<Node> ch0, Handle<Node> ch1) : type(type) {
    ch[0] = ch0;
    ch[1] = ch1;
  }
};

using NodeType = Node::Type;

// Reference counted...
class Expr {
 public:
  Handle<Node> node;

  Expr() {  // Plain input node
    node = std::make_shared<Node>(NodeType::input);
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
};

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

  std::string run(const Expr &e) {
    code = "";
    code += "{\n";
    visit(e.node);
    code += "}\n";

    return code;
  }

  void visit(const Handle<Node> &node) {
    for (auto &c : node->ch) {
      if (c)
        visit(c);
    }
    if (node->var_name == "")
      node->var_name = create_variable();
    if (node->type == NodeType::input) {
      code += fmt::format("__m256 {};\n", node->var_name);
    } else if (node->type == NodeType::add) {
      code += fmt::format("{} = {} + {};\n", node->var_name,
                          node->ch[0]->var_name, node->ch[1]->var_name);
    } else if (node->type == NodeType::mul) {
      code += fmt::format("{} = {} * {};\n", node->var_name,
                          node->ch[0]->var_name, node->ch[1]->var_name);
    } else if (node->type == NodeType::sub) {
      code += fmt::format("{} = {} - {};\n", node->var_name,
                          node->ch[0]->var_name, node->ch[1]->var_name);
    } else if (node->type == NodeType::div) {
      code += fmt::format("{} = {} / {};\n", node->var_name,
                          node->ch[0]->var_name, node->ch[1]->var_name);
    } else {
      TC_NOT_IMPLEMENTED;
    }
  }
};
}

TC_NAMESPACE_END
