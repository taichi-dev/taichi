#pragma once

#include <set>
#include "../util.h"
#include "../program.h"
#include "base.h"

TLANG_NAMESPACE_BEGIN

class Program;

class CPUCodeGen : public CodeGenBase {
 public:
  Program *prog;
  Kernel *current_kernel;
  std::map<std::string, std::string> constant_vectors;  // statement to var name
  int constant_counter;
  std::map<int, std::string> masks;

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
  }

  /*
  std::string get_mask(int group_size, int id) {
    TC_ASSERT(0 <= id && id < group_size);
    TC_ASSERT(group_size < 1000);
    int key = group_size * 1000 + id;
    if (masks.find(key) == masks.end()) {
      auto l = loop_variable(prog->current_snode);
      auto mask_name = fmt::format("mask_{}_{}", group_size, id);
      CODE_REGION(residual_begin);
      std::vector<int> offsets;
      for (int i = 0; i < simd_width; i++) {
        offsets.push_back(i / group_size + id * (simd_width / group_size));
      }
      emit_code("auto {}_indices = add(vec<int32, {}>({}), {});", mask_name,
                simd_width, l,
                get_constant(fmt::format("vec<int32, {}>({})", simd_width,
                                         vec_to_list(offsets, "{"))));
      emit_code(
          "auto {} = cmp_lt({}_indices, vec<int32, {}>({}_cache->get_n()));",
          mask_name, mask_name, simd_width,
          prog->current_snode->node_type_name);
      masks[key] = mask_name;
    }
    return masks[key];
  }
  */


  void generate_header() {
    emit_code("#include <common.h>\n");
    emit_code("#define TLANG_KERNEL\n");
    emit_code("#include \"{}\"", prog->layout_fn);
    emit_code("using namespace taichi; using namespace Tlang;");

    emit_code("extern \"C\" void " + func_name + "(Context context) {{\n");
    emit_code("auto root = ({} *)context.buffers[0];",
              prog->snode_root->node_type_name);
  }

  void generate_tail() {
    emit_code("}}\n");
  }

  void codegen(Kernel &ker);

  FunctionType compile(Program &prog, Kernel &kernel);
};

using CodeGen = CPUCodeGen;

TLANG_NAMESPACE_END
