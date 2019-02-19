//
// Created by yuanming on 2/18/19.
//

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
class GPUCodeGen : public CodeGenBase {
public:
  int simd_width;
  int group_size;

public:
  GPUCodeGen() : CodeGenBase() {
#if !defined(CUDA_FOUND)
    TC_ERROR("No GPU/CUDA support.");
#endif
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
    TC_DEBUG("Vectorizing");
    vectorized_expr.accept(*this);
    TC_DEBUG("Vectorizing");

    emit_code("}\n\n");

    emit_code("extern \"C\" void " + func_name + "(Context context) {\n");
    emit_code("auto n = context.get_range(0);\n");
    int block_size = 256;
    emit_code("{}<<<n * {} / {}, {}>>>(context);", kernel_name(), group_size,
              block_size, block_size);
    emit_code("cudaDeviceSynchronize();\n");

    emit_code("}\n");
  }

  // group_size should be batch_size here...
  FunctionType compile() {
    write_code_to_file();
    auto cmd = fmt::format(
        "nvcc {} -std=c++14 -shared -O3 -Xcompiler \"-fPIC\" --use_fast_math "
        "--ptxas-options=-allow-expensive-optimizations=true,-O3 -I {}/headers "
        "-ccbin g++-6 "
        "-D_GLIBCXX_USE_CXX11_ABI=0 -DTLANG_GPU -o {} 2> {}.log",
        get_source_fn(), get_project_fn(), get_library_fn(), get_source_fn());
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

  FunctionType get(Program &prog) {
    auto e = prog.ret;
    group_size = prog.config.group_size;
    simd_width = 32;
    TC_ASSERT(simd_width == 32);
    auto vectorized_expr = SLPVectorizer(simd_width).run(e, group_size);
    codegen(vectorized_expr, group_size);
    return compile();
  }
};




#if (0)
auto transforms = [&](Expr &ret, int group_size) {
    ret = Desugaring().run(ret);
    ret = SLPVectorizer().run(ret, group_size);
    // visualize_IR(get_source_fn() + ".slp.pdf", kernel.ret);
    if (prog->current_snode != prog->snode_root) {
      ret = LoopVectorizer().run(ret, prog->current_snode, num_groups);
    }
    AdapterPreprocessor().run(kernel, ret, group_size);
    VectorSplitter(prog->config.simd_width).run(ret);
    if (prog->config.internal_optimization)
      apply_optimizers(kernel, ret);
  };

  // transforms

  for (auto &adapter : kernel.adapters) {
    TC_ASSERT(adapter.stores);
    transforms(adapter.stores, adapter.input_group_size);
    adapter.store_exprs.resize(adapter.counter * simd_width /
                               adapter.input_group_size *
                               kernel.parallel_instances);
    // size after SLP vectorizer + Vector Splitting
  }
  TC_ASSERT(kernel.ret);
  // visualize_IR(get_source_fn() + ".scalar.pdf", prog.ret);
  // TC_P(group_size);

  transforms(kernel.ret, kernel.output_group_size);

  // body (including residual)
  for (int b = 0; b < 1 + int(has_residual); b++) {
    if (b == 1) {
      generating_residual = true;
    }
    visited.clear();
    // adapters
    CodeRegion body;
    if (b == 0) {
      body = CodeRegion::body;
    } else {
      body = CodeRegion::residual_body;
    }
    for (auto &adapter : kernel.adapters) {
      this->group_size = adapter.input_group_size;
      CODE_REGION_VAR(body);
      adapter.stores.accept(*this);
    }
    // main
    {
      this->group_size = kernel.output_group_size;
      CODE_REGION_VAR(body);
      kernel.ret.accept(*this);
    }
  }
  generating_residual = false;
#endif
