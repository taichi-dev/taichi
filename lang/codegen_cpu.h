#pragma once

#include "util.h"
#include "visitor.h"
#include "expr.h"
#include "codegen_base.h"
#include "program.h"

TLANG_NAMESPACE_BEGIN

void visualize_IR(std::string fn, Expr &expr);

class Program;

class CPUCodeGen : public CodeGenBase {
 public:
  enum class Mode : int { vv, intrinsics };
  Mode mode;
  std::string before_loop_body;
  int group_size;  // TODO: rename to current_group_size
  int simd_width;  // TODO: rename to physical simd width
  Program *prog;
  Kernel *current_kernel;
  std::map<std::string, std::string> constant_vectors;  // statement to var name
  int constant_counter;

 public:
  std::string get_constant(std::string statement) {
    if (constant_vectors.find(statement) == constant_vectors.end()) {
      CODE_REGION(exterior_shared_variable_begin);
      auto key = fmt::format("const{:04d}", constant_counter++);
      emit_code("const auto {} = {};\n", key, statement);
      constant_vectors[statement] = key;
    }
    return constant_vectors[statement];
  }

  CPUCodeGen() : CodeGenBase() {
    suffix = "cpp";
    constant_counter = 0;
    var_count = 0;
    group_size = 0;
  }

  std::string loop_variable(SNode *snode) {
    return snode->node_type_name + "_loop";
  }

  std::string index_name_local(SNode *snode, int i) {
    return fmt::format("index_{}_{}_local", snode->node_type_name, i);
  }

  std::string index_name_global(SNode *snode, int i) {
    return fmt::format("index_{}_{}_global", snode->node_type_name, i);
  }

  void generate_loop_header(SNode *snode, bool last_level = false) {
    if (snode->parent != nullptr) {
      generate_loop_header(snode->parent,
                           last_level && snode->type == SNodeType::forked);
    } else {
      return;  // no loop for root, which is a fork
    }
    auto l = loop_variable(snode);
    if (snode->parent->parent == nullptr)
      emit_code("auto {} = 0;", loop_variable(snode->parent));
    auto parent = fmt::format("{}_cache", snode->parent->node_type_name);
    emit_code("auto {}_cache = access_{}({}, {});", snode->node_type_name,
              snode->node_type_name, parent, loop_variable(snode->parent));
    if (snode->_multi_threaded) {
      auto p = snode->parent;
      while (p) {
        TC_ASSERT(!p->_multi_threaded);
        p = p->parent;
      }
      emit_code("#pragma omp parallel for");
    }
    if (last_level && snode->type != SNodeType::forked) {
      emit_code("for (int {} = 0; {} < {}_cache->get_n(); {} += {}) {{", l, l,
                snode->node_type_name, l, current_kernel->parallel_instances);
    } else {
      emit_code("for (int {} = 0; {} < {}_cache->get_n(); {} += {}) {{", l, l,
                snode->node_type_name, l, 1);
    }

    // update indices....
    for (int i = 0; i < max_num_indices; i++) {
      std::string ancester = "0 |";
      if (snode->parent->parent != nullptr) {
        ancester = index_name_global(snode->parent, i) + " |";
      }
      std::string addition = "0";
      if (snode->extractors[i].num_bits) {
        addition = fmt::format(
            "((({} >> {}) & ((1 << {}) - 1)) << {})", l,
            snode->extractors[i].dest_offset - snode->total_bit_start,
            snode->extractors[i].num_bits, snode->extractors[i].start);
      }
      emit_code("int {} = {};", index_name_local(snode, i), addition);
      emit_code("int {} = {} {};", index_name_global(snode, i), ancester,
                index_name_local(snode, i));
    }
  }

  void generate_loop_tail(SNode *snode, bool last_level = false) {
    auto l = loop_variable(snode);
    if (last_level && snode->type != SNodeType::forked) {
      // emit_code("{} += {}; b += {};", l, num_groups * unroll, unroll);
    }
    if (snode->parent != nullptr) {
      emit_code("}");
      generate_loop_tail(snode->parent,
                         last_level && snode->type == SNodeType::forked);
    } else {
      return;  // no loop for root, which is a fork
    }
  }

  void generate_header() {
    emit_code("#include <common.h>\n");
    emit_code("#define TLANG_KERNEL\n");
    emit_code("#include \"{}\"", prog->layout_fn);
    emit_code("using namespace taichi; using namespace Tlang;");

    emit_code("extern \"C\" void " + func_name + "(Context context) {\n");
    emit_code("auto {}_cache = ({} *)context.buffers[0];",
              prog->snode_root->node_type_name,
              prog->snode_root->node_type_name);

    TC_ASSERT(prog->current_snode);
    while (prog->current_snode->type == SNodeType::place) {
      prog->current_snode = prog->current_snode->parent;
      TC_ASSERT(prog->current_snode);
    }

    generate_loop_header(prog->current_snode, true);
  }

  template <typename... Args>
  void emit_code_before_loop(std::string f, Args &&... args) {
    TC_NOT_IMPLEMENTED;
  }

  void generate_tail() {
    generate_loop_tail(prog->current_snode, true);
    emit_code("}\n");
  }

  void codegen(Kernel &ker);

  template <typename T>
  static std::string vec_to_list_tmp(const std::vector<T> &val) {
    std::string members = "<";
    bool first = true;
    for (int i = 0; i < (int)val.size(); i++) {
      if (!first) {
        members += ",";
      }
      first = false;
      members += fmt::format("{}", val[i]);
    }
    members += ">";
    return members;
  }

  template <typename T>
  static std::string vec_to_list_str(const std::vector<T> &val) {
    std::string members = "{";
    bool first = true;
    for (int i = 0; i < (int)val.size(); i++) {
      if (!first) {
        members += ",";
      }
      first = false;
      members += fmt::format("{}", val[i]);
    }
    members += "}";
    return members;
  }

  void visit(Expr &expr) {
    if (mode == Mode::vv) {
      visit_vv(expr);
    } else {
      visit_intrinsics(expr);
    }
  }

  void visit_intrinsics(Expr &expr);

  void visit_vv(Expr &expr){TC_NOT_IMPLEMENTED}

  // group_size should be batch_size here...
  FunctionType compile() {
    write_code_to_file();
    auto cmd = fmt::format(
        "g++-{} {} -fopenmp -std=c++14 -shared -fPIC {} -march=native -I "
        "{}/headers -Wall "
        "-D_GLIBCXX_USE_CXX11_ABI=0 -DTLANG_CPU -o {} 2>"
        "{}.log",
        prog->config.gcc_version, get_source_fn(),
        prog->config.gcc_opt_flag(), get_project_fn(),
        get_library_fn(), get_source_fn());
    auto compile_ret = std::system(cmd.c_str());
    TC_ERROR_IF(compile_ret != 0, "Source {} compilation failed.",
                get_source_fn());
    disassemble();
    return load_function();
  }

  FunctionType get(Program &prog, Kernel &kernel);
};

using CodeGen = CPUCodeGen;

TLANG_NAMESPACE_END
