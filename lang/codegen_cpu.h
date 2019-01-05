#pragma once

#include "util.h"
#include "visitor.h"
#include "expr.h"
#include "codegen_base.h"
#include "program.h"

namespace taichi::Tlang {

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
      auto key = fmt::format("const{:04d}", constant_counter++);
      emit_code_before_loop("const auto {} = {};\n", key, statement);
      constant_vectors[statement] = key;
    }
    return constant_vectors[statement];
  }

  // Create vectorized IR for the root node
  // the vector width should be the final SIMD instruction width
  std::string get_vectorized_address(Address addr,
                                     int loop_index,
                                     int extra_offset) {
    TC_NOT_IMPLEMENTED
    return "";
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
    if (snode->type == SNodeType::forked) {
      //
    }
    if (last_level && snode->type != SNodeType::forked) {
      // emit_code("#pragma omp parallel for");
      emit_code("for (int {} = 0; {} < {}::n; {} += {}) {{", l, l,
                snode->node_type_name, l, current_kernel->parallel_instances);
    } else {
      emit_code("for (int {} = 0; {} < {}::n; {} += {}) {{", l, l,
                snode->node_type_name, l, 1);
    }
    auto parent = fmt::format("{}_cache", snode->parent->node_type_name);
    emit_code("auto {}_cache = access_{}({}, {});", snode->node_type_name,
              snode->node_type_name, parent, l);

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
    before_loop_body = code;
    code = "";

    TC_ASSERT(prog->current_snode);
    while (prog->current_snode->type == SNodeType::place) {
      prog->current_snode = prog->current_snode->parent;
      TC_ASSERT(prog->current_snode);
    }

    generate_loop_header(prog->current_snode, true);
  }

  template <typename... Args>
  void emit_code_before_loop(std::string f, Args &&... args) {
    if (sizeof...(args)) {
      before_loop_body += fmt::format(f, std::forward<Args>(args)...);
    } else {
      before_loop_body += f;
    }
  }

  void generate_tail() {
    generate_loop_tail(prog->current_snode, true);
    emit_code("}\n");
    code = before_loop_body + code;
  }

  void start_macro_loop() {
  }

  void end_macro_loop() {
    code_suffix = "\n";
  }

  std::string adapter_name(int i) {
    TC_ASSERT(i < 1000);
    return fmt::format("adapter_{:03d}", i);
  }

  /*
  T
  int num_groups,
  int num_inputs,
  int input_group_size,
  int output_group_size
  */
  std::string adapter_type(DataType dt,
                           int num_inputs,
                           int input_gs,
                           int output_gs) {
    return fmt::format("SlowAdapter<{}, {}, {}, {}, {}>", data_type_name(dt),
                       num_groups, num_inputs, input_gs, output_gs);
  }

  void create_adapter(DataType dt,
                      int i,
                      int num_inputs,
                      int input_gs,
                      int output_gs) {
    auto name = adapter_name(i);
    emit_code("{} {};", adapter_type(dt, num_inputs, input_gs, output_gs),
              adapter_name(i));
  }

  void codegen(Kernel &ker);

  std::string vv_type_str(int width, DataType data_type) {
    return fmt::format("VV<{}, {}>", width, data_type_name(data_type));
  }

  std::string vv_constant_str(int width, DataType data_type, int64 val) {
    return fmt::format("VV<{}, {}>({})", width, data_type_name(data_type), val);
  }

  std::string vv_constant_str(int width, DataType data_type, float32 val) {
    return fmt::format("VV<{}, {}>({})", width, data_type_name(data_type), val);
  }

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

  template <typename T>
  static std::string vv_constant_str(int width,
                                     DataType data_type,
                                     const std::vector<T> &val) {
    return fmt::format("VV<{}, {}>({})", width, data_type_name(data_type),
                       vec_to_list_str(val));
  }

  void visit(Expr &expr) {
    if (mode == Mode::vv) {
      visit_vv(expr);
    } else {
      visit_intrinsics(expr);
    }
  }

  void visit_intrinsics(Expr &expr);

  void visit_vv(Expr &expr);

  // group_size should be batch_size here...
  FunctionType compile() {
    write_code_to_file();
    auto cmd = fmt::format(
        "g++-7 {} -fopenmp -std=c++14 -shared -fPIC -O3 -march=native -I "
        "{}/headers -Wall "
        "-D_GLIBCXX_USE_CXX11_ABI=0 -DTLANG_CPU -o {} 2>"
        "{}.log",
        get_source_fn(), get_project_fn(), get_library_fn(), get_source_fn());
    auto compile_ret = std::system(cmd.c_str());
    TC_ASSERT(compile_ret == 0);
    disassemble();
    return load_function();
  }

  FunctionType get(Program &prog, Kernel &kernel);
};

using CodeGen = CPUCodeGen;
}  // namespace taichi::Tlang
