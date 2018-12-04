#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <dlfcn.h>
#include <set>

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

class Expr;

// TODO: do we need polymorphism here?
class Node {
 public:
  enum class Type : int { mul, add, sub, div, load, store, combine, constant };

  Address addr;
  std::vector<Expr> ch;       // Four child max
  std::vector<Expr> members;  // for vectorized instructions
  Type type;
  std::string var_name;
  float64 value;
  bool is_vectorized;

  Node(Type type) : type(type) {
    is_vectorized = false;
  }

  Node(Type type, Expr ch0, Expr ch1);

  int member_id(const Expr &expr) const;
};

using NodeType = Node::Type;

// Reference counted...
class Expr {
 private:
  Handle<Node> node;

 public:
  Expr() {
  }

  Expr(float64 val) {
    // cretea a constant node
    node = std::make_shared<Node>(NodeType::constant);
    node->value = val;
  }

  Expr(Handle<Node> node) : node(node) {
  }

  template <typename... Args>
  static Expr create(Args &&... args) {
    return Expr(std::make_shared<Node>(std::forward<Args>(args)...));
  }

#define BINARY_OP(op, name)                            \
  Expr operator op(const Expr &o) const {              \
    return Expr::create(NodeType::name, node, o.node); \
  }

  BINARY_OP(*, mul);
  BINARY_OP(+, add);
  BINARY_OP(-, sub);
  BINARY_OP(/, div);
#undef BINARY_OP

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

  Node *operator->() {
    return node.get();
  }

  const Node *operator->() const {
    return node.get();
  }

  bool operator<(const Expr &o) const {
    return node.get() < o.node.get();
  }

  operator bool() const {
    return node.get() != nullptr;
  }

  operator void *() const {
    return (void *)node.get();
  }

  bool operator==(const Expr &o) const {
    return (void *)(*this) == (void *)o;
  }
};

Node::Node(Type type, Expr ch0, Expr ch1) : Node(type) {
  ch.resize(2);
  ch[0] = ch0;
  ch[1] = ch1;
}

inline Expr load(Address addr) {
  auto n = std::make_shared<Node>(NodeType::load);
  TC_ASSERT(addr.initialized());
  n->addr = addr;
  TC_ASSERT(0 <= addr.stream_id && addr.stream_id < 3);
  return Expr(n);
}

int Node::member_id(const Expr &expr) const {
  for (int i = 0; i < members.size(); i++) {
    if (members[i] == expr) {
      return i;
    }
  }
  return -1;
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
  int group_size;
  int num_groups;
  int id;
  std::map<NodeType, std::string> binary_ops;
  std::string folder;

  CodeGen(Mode mode = Mode::vector, int simd_width = 8)
      : var_count(0), mode(mode), simd_width(simd_width) {
  }

  std::string create_variable() {
    TC_ASSERT(var_count < 10000);
    return fmt::format("var_{:04d}", var_count++);
  }

  using FunctionType = void (*)(float32 *, float32 *, float32 *, int);

  std::string run(Expr &expr, int group_size = 1) {
    TC_ASSERT(mode == Mode::vector);
    // group_size = expr->ch.size();
    this->group_size = group_size;
    num_groups = simd_width / group_size;
    TC_WARN_IF(simd_width % group_size != 0, "insufficient lane usage");

    id = get_code_gen_id();
    func_name = fmt::format("func{:06d}", id);
    binary_ops[NodeType::add] = "+";
    binary_ops[NodeType::sub] = "-";
    binary_ops[NodeType::mul] = "*";
    binary_ops[NodeType::div] = "/";
    folder = "_tlang_cache/";
    create_directories(folder);
    code = "#include <immintrin.h>\n#include <cstdio>\n";
    code += "using float32 = float;\n";
    code += "using float64 = double;\n\n";
    code += "extern \"C\" void " + func_name +
            "(float32 *stream00, float32 *stream01, float32 "
            "*stream02, "
            "int n) {\n";
    code += fmt::format("for (int i = 0; i < n; i += {}) {{\n", num_groups);
    auto vectorized_expr = vectorize(expr, group_size, num_groups);
    code_gen(vectorized_expr);
    code += "}\n}\n";
    return code;
  }

  std::string get_scalar_suffix(int i) {
    return fmt::format("_{:03d}", i);
  }

  std::vector<Expr> reachable_exprs(Expr &expr) {
    std::vector<Expr> ret;
    std::set<Expr> visited;

    std::function<void(Expr &)> visit = [&](Expr &expr) {
      if (visited.find(expr) != visited.end())
        return;
      visited.insert(expr);
      ret.push_back(expr);
      for (auto c : expr->ch) {
        code_gen(c);
      }
    };
    code_gen(expr);
    return ret;
  }

  Expr repeat(Expr &expr, int offset) {
    TC_NOT_IMPLEMENTED
    std::set<Expr> visited;
    Expr new_expr;

    // Copy with duplication detection

    return new_expr;
  }

  std::map<Expr, Expr> scalar_to_vector;
  // Create vectorized IR
  // the vector width should be the final SIMD instruction width
  Expr vectorize(Expr &expr, int group_size, int num_groups) {
    TC_ASSERT(group_size * num_groups == simd_width);
    scalar_to_vector.clear();
    // expr should be a ret Op, with its children store Ops.
    // The stores are repeated by a factor of 'pack_size'
    TC_ASSERT(expr->ch.size() == group_size);
    TC_ASSERT(expr->type == NodeType::combine);
    // Create the root group
    auto root = Expr::create(NodeType::store);
    root->is_vectorized = true;
    for (int i = 0; i < group_size; i++) {
      auto ch = expr->ch[i];
      TC_ASSERT(ch->type == NodeType::store);
      root->members.push_back(ch);
      if (i > 0) {
        TC_ASSERT(prior_to(root->members[i - 1], root->members[i]));
      }
    }
    vectorize(root);
    return root;
  }

  void vectorize(Expr &expr) {
    // Note: expr may be replaced by an existing vectorized Expr
    if (scalar_to_vector.find(expr->members[0]) != scalar_to_vector.end()) {
      auto existing = scalar_to_vector[expr->members[0]];
      TC_ASSERT(existing->members.size() == expr->members.size());
      for (int i = 0; i < existing->members.size(); i++) {
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
        for (int i = 0; i < member->ch.size(); i++) {
          vectorized_children[i].push_back(member->ch[i]);
        }
      } else {
        TC_ASSERT(type == member->type);
        TC_ASSERT(vectorized_children.size() == member->ch.size());
      }
    }

    auto vectorized_expr = Expr::create(type);
    vectorized_expr->is_vectorized = true;

    for (int i = 0; i < vectorized_children.size(); i++) {
      auto ch = Expr::create(vectorized_children[i][0]->type);
      ch->members = vectorized_children[i];
      expr->ch.push_back(ch);
      vectorize(ch);
    }

    vectorized_expr->addr = expr->members[0]->addr;
    if (vectorized_expr->addr.coeff_aosoa_group_size == 0) {
      vectorized_expr->addr.coeff_aosoa_group_size = group_size;
      vectorized_expr->addr.coeff_aosoa_stride = 0;
    }
    expr = vectorized_expr;
  }

  /*
  void vectorized_codegen(Expr &expr) {
    for (auto &c: expr->ch) {
      if (c) {
        vectorized_codegen(c);
      }
    }
    if (expr->var_name == "") {
      expr->var_name = create_variable();
    } else
      return; // visited
  }
  */

  std::string get_vectorized_address(Address addr) {
    auto stream_name = fmt::format("stream{:02d}", addr.stream_id);
    auto stride = addr.coeff_i + group_size / addr.coeff_aosoa_group_size *
                                     addr.coeff_aosoa_stride;
    auto offset = addr.coeff_const;
    return fmt::format("(&{}[{} * n + {} * i + {}])\n", stream_name,
                       addr.coeff_imax, stride, offset);
  }

  void code_gen(Expr &expr) {
    TC_ASSERT(expr->is_vectorized);
    for (auto &c : expr->ch) {
      if (c)
        code_gen(c);
    }
    if (expr->var_name == "")
      expr->var_name = create_variable();
    else
      return;  // visited
    if (binary_ops.find(expr->type) != binary_ops.end()) {
      auto op = binary_ops[expr->type];
      if (mode == Mode::vector) {
        code += fmt::format("auto {} = {} {} {};\n", expr->var_name,
                            expr->ch[0]->var_name, op, expr->ch[1]->var_name);
      } else if (mode == Mode::scalar) {
        for (int i = 0; i < simd_width; i++) {
          auto suf = get_scalar_suffix(i);
          code += fmt::format("auto {} = {} {} {};\n", expr->var_name + suf,
                              expr->ch[0]->var_name + suf, op,
                              expr->ch[1]->var_name + suf);
        }
      }
    } else if (expr->type == NodeType::load) {
      auto stream_name = fmt::format("stream{:02d}", expr->addr.stream_id);

      if (mode == Mode::vector) {
        for (int i = 0; i + 1 < (int)expr->members.size(); i++) {
          TC_ASSERT(prior_to(expr->members[i], expr->members[i + 1]));
        }
        auto addr = expr->addr;
        auto i_stride = num_groups;
        TC_ASSERT(i_stride == addr.coeff_aosoa_group_size);
        // TC_ASSERT(expr->members[0]->addr.coeff_i);
        std::string load_instr =
            simd_width == 8 ? "_mm256_load_ps" : "_mm512_load_ps";
        code +=
            fmt::format("auto {} = {}({});\n", get_vectorized_address(addr));
      } else {
        TC_NOT_IMPLEMENTED
        for (int i = 0; i < simd_width; i++) {
          auto suf = get_scalar_suffix(i);
          code += fmt::format("auto {} = {}[{} * i + {} + {}];\n",
                              expr->var_name + suf, stream_name,
                              expr->addr.coeff_i, expr->addr.coeff_const, i);
        }
      }
    } else if (expr->type == NodeType::store) {
      auto stream_name = fmt::format("stream{:02d}", expr->addr.stream_id);
      if (mode == Mode::vector) {
        std::string store_instr =
            simd_width == 8 ? "_mm256_store_ps" : "_mm512_store_ps";
        code += fmt::format("{}({});\n", store_instr,
                            get_vectorized_address(expr->addr));
      } else {
        TC_NOT_IMPLEMENTED
        for (int i = 0; i < simd_width; i++) {
          auto suf = get_scalar_suffix(i);
          code += fmt::format("{}[{} * i + {} + {}] = {};\n", stream_name,
                              expr->addr.coeff_i, expr->addr.coeff_const, i,
                              expr->ch[0]->var_name + suf);
        }
      }
    } else if (expr->type == NodeType::combine) {
      // do nothing
    } else {
      TC_P((int)expr->type);
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

  FunctionType get(Expr &e, int group_size = 4) {
    // SLP(e, group_size);
    run(e);
    {
      std::ofstream of(get_source_fn());
      of << code;
    }
    std::system(fmt::format("clang-format -i {}", get_source_fn()).c_str());
    auto cmd = fmt::format(
        "g++ {} -std=c++14 -shared -fPIC -O3 -march=native "
        "-D_GLIBCXX_USE_CXX11_ABI=0 -o {}",
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
    auto address1 = a->addr;
    auto address2 = b->addr;
    return address1.same_type(address2) &&
           address1.offset() + 1 == address2.offset();
  }

  std::vector<Expr> extract_instructions(Expr root_expr) {
    std::vector<Expr> inst;
    std::set<void *> visited;

    std::function<void(Expr)> walk = [&](Expr expr) -> void {
      TC_ASSERT(expr);
      if (visited.find(expr) != visited.end())
        return;
      visited.insert(expr);
      for (auto &ch : expr->ch) {
        walk(ch);
      }
      inst.push_back(expr);
    };

    walk(root_expr);

    return inst;
  }

  std::vector<Expr> inst;
  std::vector<std::vector<int>> groups;
  std::vector<bool> grouped;

  std::vector<int> continuous_loads(int i) {
    std::vector<int> ret;
    if (grouped[i] || inst[i]->type != NodeType::load) {
      return ret;
    }
    ret.push_back(i);
    while (1) {
      bool found = false;
      for (int j = 0; j < inst.size(); j++) {
        if (grouped[j] || i == j || inst[i]->type != NodeType::load) {
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

      // Extend
      if (inst_with_max_length != -1) {
        TC_P(C);
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

/*
 Expr should be what the users play with.
   Simply a ref-counted pointer to nodes, with some operator overloading for
 users to program Node is the IR node, with computational graph connectivity,
 imm, op type etc.

 No double support this time.
 */
