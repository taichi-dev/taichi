#pragma once

#include <set>
#include "util.h"
#include "codegen_base.h"
#include "program.h"

TLANG_NAMESPACE_BEGIN

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
  bool has_residual;
  bool generating_residual;
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
    var_count = 0;
    group_size = 0;
    has_residual = false;
    generating_residual = false;
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

  void generate_loop_header(SNode *snode, bool last_level = false) {
#if (0)
    if (snode->parent != nullptr) {
      generate_loop_header(snode->parent,
                           last_level && snode->type == SNodeType::forked);
    } else {
      return;  // no loop for root, which is a fork
    }
    auto l = loop_variable(snode);
    bool interior = last_level && snode->type != SNodeType::forked;
    CodeRegion r;
    if (last_level)
      r = CodeRegion::interior_loop_begin;
    else
      r = CodeRegion::exterior_loop_begin;
    CODE_REGION_VAR(r);
    if (snode->parent->parent == nullptr)
      emit_code("auto {} = 0;", loop_variable(snode->parent));
    auto parent = fmt::format("{}_cache", snode->parent->node_type_name);
    emit_code("auto {}_cache = access_{}({}, {});", snode->node_type_name,
              snode->node_type_name, parent, loop_variable(snode->parent));
    emit_code("int {};", l);

    if (snode->type == SNodeType::pointer) {
      emit_code("if (!{}_cache->data) continue;", snode->node_type_name, l);
    }

    if (snode->type != SNodeType::hashed) {
      emit_code("auto {}_cache_n = {}_cache->get_n();", snode->node_type_name,
                snode->node_type_name);
    }
    if (snode->_multi_threaded) {
      auto p = snode->parent;
      while (p) {
        TC_ASSERT(!p->_multi_threaded);
        p = p->parent;
      }
      emit_code("#pragma omp parallel for");
    }
    if (interior) {
      if (!has_residual) {
        emit_code("for ({} = 0; {} < {}_cache_n; {} += {}) {{", l, l,
                  snode->node_type_name, l, current_kernel->parallel_instances);
      } else {
        int residual = current_kernel->parallel_instances >
                               1  // when only one instance, no residual loop.
                           ? 0
                           : current_kernel->parallel_instances;
        emit_code("for ({} = 0; {} + {} < {}_cache_n; {} += {}) {{", l, l,
                  residual, snode->node_type_name, l,
                  current_kernel->parallel_instances

        );
      }
    } else {
      if (snode->type == SNodeType::hashed) {
        emit_code("for (auto &{}_it : {}_cache->data) {{", l,
                  snode->node_type_name);
        emit_code("int {} = {}_it.first;", l, l);
      } else {
        emit_code("for ({} = 0; {} < {}_cache_n; {} += {}) {{", l, l,
                  snode->node_type_name, l, 1);
      }
    }

    if (has_residual && last_level) {
      CODE_REGION(residual_begin);  // TODO: DRY..
      emit_code("if ({} < {}_cache_n) {{", l, snode->node_type_name);
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
      if (has_residual && last_level) {
        CODE_REGION(residual_begin);  // TODO: DRY..
        emit_code("int {} = {};", index_name_local(snode, i), addition);
        emit_code("int {} = {} {};", index_name_global(snode, i), ancester,
                  index_name_local(snode, i));
      }
    }
    if (has_residual && last_level) {
      CODE_REGION(residual_end);
      emit_code("}}");
    }
#endif
  }

  void generate_loop_tail(SNode *snode, bool last_level = false) {
    CodeRegion r;
    r = CodeRegion::exterior_loop_end;
    auto l = loop_variable(snode);
    if (last_level && snode->type != SNodeType::forked) {
      // emit_code("{} += {}; b += {};", l, num_groups * unroll, unroll);
      r = CodeRegion::interior_loop_end;
    }
    CODE_REGION_VAR(r);
    if (snode->parent != nullptr) {
      CODE_REGION_VAR(last_level ? CodeRegion::interior_loop_end
                                 : CodeRegion::exterior_loop_end);
      emit_code("}}\n");
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

    emit_code("extern \"C\" void " + func_name + "(Context context) {{\n");
    emit_code("auto root = ({} *)context.buffers[0];",
              prog->snode_root->node_type_name);
  }

  void generate_tail() {
    // generate_loop_tail(prog->current_snode, true);
    emit_code("}}\n");
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

  FunctionType compile();

  FunctionType get(Program &prog, Kernel &kernel);
};

using CodeGen = CPUCodeGen;

TLANG_NAMESPACE_END
