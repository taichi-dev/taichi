#pragma once

#include "util.h"
#include "visitor.h"
#include "expr.h"
#include "codegen_base.h"

namespace taichi::Tlang {

void visualize_IR(std::string fn, Expr &expr);

class Program;

class CPUCodeGen : public CodeGenBase {
 public:
  int unroll;
  int prefetch;
  enum class Mode : int { vv, intrinsics };
  Mode mode;
  int simd_width;
  int group_size;
  std::string before_loop_body;
  Program *prog;
  std::map<std::string, std::string> constant_vectors; // statement to var name
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
    TC_ASSERT(addr.buffer_id != -1);
    auto buffer_name =
        fmt::format("context.get_buffer<float32>({:02d})", addr.buffer_id);
    auto stride = addr.coeff_i * num_groups;
    if (addr.coeff_aosoa_group_size != 0) {
      stride +=
          num_groups / addr.coeff_aosoa_group_size * addr.coeff_aosoa_stride;
    }
    auto offset = addr.coeff_const;
    return fmt::format("&{}[{} * n + {} * (b + {}) + {} + {}]", buffer_name,
                       addr.coeff_imax, stride, loop_index, offset,
                       extra_offset);
  }

  CPUCodeGen() : CodeGenBase() {
    suffix = "cpp";
    prefetch = 0;
    unroll = 1;
    constant_counter = 0;
    var_count = 0;
  }

  void generate_header() {
    emit_code(
        "#include <common.h>\n using namespace taichi; using namespace Tlang;");
    emit_code("extern \"C\" void " + func_name + "(Context context) {\n");
    emit_code("auto n = context.get_range(0);\n\n");
    before_loop_body = code;
    code = "";
    emit_code("for (int i = 0, b = 0; (unsigned)i < n; ) {{\n", num_groups);
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
    emit_code("}\n}\n");
    code = before_loop_body + code;
  }

  void start_macro_loop() {
    // code_suffix = " \\\n";
    // emit_code("#define LOOP(loop_index) {");
  }

  void end_macro_loop() {
    emit_code("i += {}; b += {};", num_groups * unroll, unroll);
    code_suffix = "\n";
    // emit_code("}\n");
    for (int i = 0; i < unroll; i++) {
      // emit_code("LOOP({});", i);
    }
    // emit_code("#undef LOOP\n");
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

  void codegen(Program &prog, int group_size);

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
  static std::string vec_to_list_tmp(std::vector<T> val) {
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
  static std::string vec_to_list_str(std::vector<T> val) {
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

  std::string vv_constant_str(int width,
                              DataType data_type,
                              std::vector<int> val) {
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
        "g++ {} -std=c++14 -shared -fPIC -O3 -march=native -I {}/headers -Wall "
        "-D_GLIBCXX_USE_CXX11_ABI=0 -DTLANG_CPU -o {}",
        get_source_fn(), get_project_fn(), get_library_fn());
    auto compile_ret = std::system(cmd.c_str());
    TC_ASSERT(compile_ret == 0);
#if defined(TC_PLATFORM_LINUX)
    auto objdump_ret = system(
        fmt::format("objdump {} -d > {}.s", get_library_fn(), get_library_fn())
            .c_str());
    trash(objdump_ret);
#endif
    return load_function();
  }

  FunctionType get(Program &prog);
};

using CodeGen = CPUCodeGen;
}
