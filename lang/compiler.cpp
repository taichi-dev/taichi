#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "tlang.h"

TC_NAMESPACE_BEGIN

namespace Tlang {

Program *current_program = nullptr;

class CPUCodeGen : public CodeGenBase {
 public:
  int unroll;
  int prefetch;
  enum class Mode : int { scalar, vector };
  Mode mode;
  int simd_width;
  int group_size;

 public:
  // Create vectorized IR for the root node
  // the vector width should be the final SIMD instruction width
  std::string get_vectorized_address(Address addr, int extra_offset = 0) {
    TC_ASSERT(addr.buffer_id != -1);
    auto buffer_name =
        fmt::format("context.get_buffer<float32>({:02d})", addr.buffer_id);
    auto stride =
        addr.coeff_i * num_groups +
        num_groups / addr.coeff_aosoa_group_size * addr.coeff_aosoa_stride;
    auto offset = addr.coeff_const;
    return fmt::format("&{}[{} * n + {} * (g + loop_index) + {} + {}]",
                       buffer_name, addr.coeff_imax, stride, offset,
                       extra_offset);
  }

  CPUCodeGen() : CodeGenBase() {
    suffix = "cpp";
    prefetch = 0;
    unroll = 1;
    var_count = 0;
  }

  void codegen(Expr &vectorized_expr, int group_size = 1) {
    TC_ASSERT(mode == Mode::vector);
    this->group_size = group_size;
    TC_ASSERT(group_size != 0);
    // group_size = expr->ch.size();
    num_groups = simd_width / group_size;
    TC_WARN_IF(simd_width % group_size != 0, "insufficient lane usage");

    emit_code(
        "#include <common.h>\n using namespace taichi; using namespace Tlang;");
    emit_code("extern \"C\" void " + func_name + "(Context context) {\n");
    emit_code("#define LOOP(loop_index) {\\\n");

    // Body
    vectorized_expr.accept(*this);

    emit_code("}\n");
    emit_code("auto n = context.get_range(0);\n");
    emit_code("for (int i = 0, g = 0; i < n; ) {{\n", num_groups);
    for (int i = 0; i < unroll; i++) {
      emit_code("LOOP({});", i);
    }
    emit_code("i += {}; g += {};", num_groups * unroll, unroll);
    emit_code("}\n}\n");
    emit_code("#undef LOOP");
  }

  void visit(Expr &expr) override {
    TC_ASSERT(expr->is_vectorized);
    TC_ASSERT(expr->members.size() == 0 ||
              (int)expr->members.size() == group_size);
    // TC_P(expr->ch.size());
    if (expr->var_name == "")
      expr->var_name = create_variable();
    else
      return;  // visited
    if (binary_ops.find(expr->type) != binary_ops.end()) {
      auto op = binary_ops[expr->type];
      if (mode == Mode::vector) {
        emit_code("auto {} = {} {} {}; \\\n", expr->var_name,
                  expr->ch[0]->var_name, op, expr->ch[1]->var_name);
      } else if (mode == Mode::scalar) {
        for (int i = 0; i < simd_width; i++) {
          auto suf = get_scalar_suffix(i);
          emit_code("auto {} = {} {} {}; \\\n", expr->var_name + suf,
                    expr->ch[0]->var_name + suf, op,
                    expr->ch[1]->var_name + suf);
        }
      }
    } else if (expr->type == NodeType::load) {
      auto buffer_name = fmt::format("buffer{:02d}", expr->addr.buffer_id);

      if (mode == Mode::vector) {
        // TC_P(expr->members.size());
        std::vector<int> offsets;
        for (int i = 0; i + 1 < (int)expr->members.size(); i++) {
          TC_ASSERT(
              expr->members[i]->addr.same_type(expr->members[i + 1]->addr));
        }
        for (int i = 0; i < (int)expr->members.size(); i++) {
          offsets.push_back(expr->members[i]->addr.offset());
        }
        auto addr = expr->addr;
        auto i_stride = num_groups;
        // TC_P(i_stride);
        // TC_P(addr.coeff_aosoa_group_size);
        TC_ASSERT(i_stride == addr.coeff_aosoa_group_size);
        // TC_ASSERT(expr->members[0]->addr.coeff_i);
        std::string load_instr =
            simd_width == 8 ? "_mm256_load_ps" : "_mm512_load_ps";
        bool needs_shuffle = false;
        if (addr.coeff_const % simd_width != 0) {
          addr.coeff_const -= addr.coeff_const % simd_width;
          needs_shuffle = true;
        }
        if (prefetch != 0) {
          // https://stackoverflow.com/questions/46521694/what-are-mm-prefetch-locality-hints
          emit_code("if (loop_index == 0) _mm_prefetch({}, _MM_HINT_NTA); \\\n",
                    get_vectorized_address(addr, prefetch));
        }
        emit_code("auto {}_immediate = {}({}); \\\n", expr->var_name,
                  load_instr, get_vectorized_address(addr));
        auto emit_shuffle = [&](std::string imm) {
          emit_code(
              "auto {} = _mm256_shuffle_ps({}_immediate, {}_immediate, "
              "{});\\\n",
              expr->var_name, expr->var_name, expr->var_name, imm);
          needs_shuffle = false;
        };
        if (group_size == 1) {
          emit_code("auto {} = {}_immediate; \\\n", expr->var_name,
                    expr->var_name);
        } else {
          TC_ASSERT(group_size <= 4);
          // detect patterns
          int offset_const = offsets[0] % simd_width;
          int offset_inc = offsets[1] - offsets[0];
          if (group_size == 2) {
            if (offset_const == 0 && offset_inc == 1) {
              emit_code("auto {} = {}_immediate; \\\n", expr->var_name,
                        expr->var_name);
            } else if (offset_inc == 0) {
              if (offset_const == 0) {
                emit_shuffle("0xA0");
              } else if (offset_const == 1) {
                emit_shuffle("0xF5");
              } else {
                TC_NOT_IMPLEMENTED;
              }
            } else {
              TC_P(offset_const);
              TC_P(offset_inc);
              TC_NOT_IMPLEMENTED;
            }
          } else if (group_size == 4) {
            if (offset_const == 0 && offset_inc == 1) {
              emit_code("auto {} = {}_immediate;\\\n", expr->var_name,
                        expr->var_name);
            } else if (offset_inc == 0) {
              if (offset_const == 0) {
                emit_shuffle("0x00");
              } else if (offset_const == 1) {
                emit_shuffle("0x55");
              } else if (offset_const == 2) {
                emit_shuffle("0xAA");
              } else if (offset_const == 3) {
                emit_shuffle("0xFF");
              } else {
                TC_NOT_IMPLEMENTED;
              }
            } else {
              TC_P(offset_const);
              TC_P(offset_inc);
              TC_NOT_IMPLEMENTED;
            }

          } else {
            TC_NOT_IMPLEMENTED
          }
          TC_ASSERT(needs_shuffle == false);
        }

      } else {
        TC_NOT_IMPLEMENTED
        for (int i = 0; i < simd_width; i++) {
          auto suf = get_scalar_suffix(i);
          emit_code("auto {} = {}[{} * i + {} + {}];\\\n", expr->var_name + suf,
                    buffer_name, expr->addr.coeff_i, expr->addr.coeff_const, i);
        }
      }
    } else if (expr->type == NodeType::store) {
      auto buffer_name = fmt::format("buffer{:02d}", expr->addr.buffer_id);
      if (mode == Mode::vector) {
        std::string store_instr =
            // simd_width == 8 ? "_mm256_stream_ps" : "_mm512_stream_ps";
            simd_width == 8 ? "_mm256_store_ps" : "_mm512_store_ps";
        emit_code("{}({}, {}); \\\n", store_instr,
                  get_vectorized_address(expr->addr), expr->ch[0]->var_name);
      } else {
        TC_NOT_IMPLEMENTED
        for (int i = 0; i < simd_width; i++) {
          auto suf = get_scalar_suffix(i);
          emit_code("{}[{} * i + {} + {}] = {}; \\\n", buffer_name,
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

  // group_size should be batch_size here...
  FunctionType compile() {
    write_code_to_file();
    auto cmd = fmt::format(
        "g++ {} -std=c++14 -shared -fPIC -O3 -march=native -I {}/headers "
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

  FunctionType get(Expr &e,
                   int group_size,
                   CPUCodeGen::Mode mode = CPUCodeGen::Mode::vector,
                   int simd_width = 8) {
    this->mode = mode;
    this->simd_width = simd_width;
    auto vectorized_expr = Vectorizer(simd_width).run(e, group_size);
    codegen(vectorized_expr, group_size);
    return compile();
  }
};

using CodeGen = CPUCodeGen;

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
class GPUCodeGen : public CodeGenBase {
 public:
  int simd_width;
  int group_size;

 public:
  GPUCodeGen() : CodeGenBase() {
    suffix = "cu";
    simd_width = 32;
  }

  std::string kernel_name() {
    return fmt::format("{}_kernel", func_name);
  }

  void codegen(Expr &vectorized_expr, int group_size = 1) {
    this->group_size = group_size;
    TC_ASSERT(group_size != 0);
    // group_size = expr->ch.size();
    num_groups = simd_width / group_size;
    TC_WARN_IF(simd_width % group_size != 0, "insufficient lane usage");

    emit_code(
        "#include <common.h>\n using namespace taichi; using namespace "
        "Tlang;\n\n");

    emit_code("__global__ void {}(Context context) {{", kernel_name());
    emit_code("auto n = context.get_range(0);\n");
    emit_code("auto linear_idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
    emit_code("if (linear_idx >= n * {}) return;\n", group_size);
    emit_code("auto g = linear_idx / {} / ({} / {}); \n", group_size,
              simd_width, group_size);
    emit_code("auto sub_g_id = linear_idx / {} % ({} / {}); \n", group_size,
              simd_width, group_size);
    emit_code("auto g_idx = linear_idx % {}; \n", group_size);
    emit_code("int l = threadIdx.x & 0x1f;\n");
    emit_code("int loop_index = 0;");

    // Body
    vectorized_expr.accept(*this);

    emit_code("}\n\n");

    emit_code("extern \"C\" void " + func_name + "(Context context) {\n");
    emit_code("auto n = context.get_range(0);\n");
    int block_size = 256;
    emit_code("{}<<<n * {} / {}, {}>>>(context);", kernel_name(), group_size,
              block_size, block_size);
    emit_code("cudaDeviceSynchronize();\n");

    emit_code("}\n");
  }

  void visit(Expr &expr) override {
    TC_ASSERT(expr->is_vectorized);
    TC_ASSERT(expr->members.size() == 0 ||
              (int)expr->members.size() == group_size);
    // TC_P(expr->ch.size());
    if (expr->var_name == "")
      expr->var_name = create_variable();
    else
      return;  // visited
    if (binary_ops.find(expr->type) != binary_ops.end()) {
      auto op = binary_ops[expr->type];
      emit_code("auto {} = {} {} {}; \\\n", expr->var_name,
                expr->ch[0]->var_name, op, expr->ch[1]->var_name);
    } else if (expr->type == NodeType::load) {
      auto buffer_name = fmt::format("buffer{:02d}", expr->addr.buffer_id);
      std::vector<int> offsets;
      for (int i = 0; i + 1 < (int)expr->members.size(); i++) {
        TC_ASSERT(expr->members[i]->addr.same_type(expr->members[i + 1]->addr));
      }
      for (int i = 0; i < (int)expr->members.size(); i++) {
        offsets.push_back(expr->members[i]->addr.offset());
      }
      auto addr = expr->addr;
      auto i_stride = num_groups;
      TC_ASSERT(i_stride == addr.coeff_aosoa_group_size);
      if (addr.coeff_const % simd_width != 0) {
        addr.coeff_const -= addr.coeff_const % simd_width;
      }

      if (group_size > 1) {
        // detect patterns
        int offset_const = offsets[0] % simd_width;
        int offset_inc = offsets[1] - offsets[0];
        for (int i = 0; i + 1 < (int)offsets.size(); i++) {
          TC_ASSERT(offset_inc == offsets[i + 1] - offsets[i]);
        }
        emit_code("auto {} = {}; \\\n", expr->var_name,
                  get_vectorized_address(addr, offset_const, offset_inc));
      } else {
        emit_code("auto {} = {}; \\\n", expr->var_name,
                  get_vectorized_address(addr));
      }
    } else if (expr->type == NodeType::store) {
      emit_code("{} = {}; \\\n", get_vectorized_address(expr->addr),
                expr->ch[0]->var_name);
      // emit_code("printf(\"%f\\n\", {}); \\\n", expr->ch[0]->var_name);
      // emit_code("printf(\"%d <- %f\\n\", {}, {}); \\\n",
      //        get_vectorized_index(expr->addr), expr->ch[0]->var_name);
    } else if (expr->type == NodeType::combine) {
      // do nothing
    } else {
      TC_P((int)expr->type);
      TC_NOT_IMPLEMENTED;
    }
  }

  // group_size should be batch_size here...
  FunctionType compile() {
    write_code_to_file();
    auto cmd = fmt::format(
        "nvcc {} -std=c++14 -shared -O3 -Xcompiler \"-fPIC\" --use_fast_math "
        "--ptxas-options=-allow-expensive-optimizations=true,-O3 -I {}/headers "
        "-ccbin g++-6 "
        "-D_GLIBCXX_USE_CXX11_ABI=0 -DTLANG_GPU -o {}",
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

  FunctionType get(Expr &e,
                   int group_size,
                   CPUCodeGen::Mode mode = CPUCodeGen::Mode::vector,
                   int simd_width = 32) {
    TC_ASSERT(simd_width == 32);
    auto vectorized_expr = Vectorizer(simd_width).run(e, group_size);
    codegen(vectorized_expr, group_size);
    return compile();
  }

  // Create vectorized IR for the root node
  // the vector width should be the final SIMD instruction width
  std::string get_vectorized_address(Address addr,
                                     int extra_offset = 0,
                                     int g_idx_inc = 1) {
    TC_ASSERT(addr.buffer_id != -1);
    auto buffer_name =
        fmt::format("context.get_buffer<float32>({:02d})", addr.buffer_id);
    auto warp_stride =
        addr.coeff_i * num_groups +
        num_groups / addr.coeff_aosoa_group_size * addr.coeff_aosoa_stride;
    auto offset = addr.coeff_const;
    return fmt::format(
        "{}[{} * n + {} * (g + loop_index) + sub_g_id * {} + {} + {} + g_idx * "
        "{}]",
        buffer_name, addr.coeff_imax, warp_stride, addr.coeff_i, offset,
        extra_offset, (g_idx_inc));
  }

  /*
  std::string get_vectorized_index(Address addr,
                                   int extra_offset = 0,
                                   int g_idx_inc = 1) {
    TC_ASSERT(addr.buffer_id != -1);
    auto buffer_name =
        fmt::format("context.get_buffer<float32>({:02d})", addr.buffer_id);
    auto warp_stride =
        addr.coeff_i * num_groups +
        num_groups / addr.coeff_aosoa_group_size * addr.coeff_aosoa_stride;
    auto offset = addr.coeff_const;
    return fmt::format(
        "{} * n + {} * (g + loop_index) + sub_g_id * {} + {} + {} + g_idx * "
        "{}",
        addr.coeff_imax, warp_stride, addr.coeff_i, offset, extra_offset,
        g_idx_inc);
  }
  */
};

void Program::compile() {
  alloc.materialize();
  TC_ASSERT(config.simd_width > 0);
  TC_ASSERT(config.group_size > 0);
  if (config.arch == CompileConfig::Arch::x86_64) {
    CPUCodeGen codegen;
    function = codegen.get(ret, config.group_size);
  } else if (config.arch == CompileConfig::Arch::gpu) {
    GPUCodeGen codegen;
    function = codegen.get(ret, config.group_size);
  } else {
    TC_NOT_IMPLEMENTED;
  }
}
}

TC_NAMESPACE_END
