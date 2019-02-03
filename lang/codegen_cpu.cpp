#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "util.h"
#include "codegen_cpu.h"
#include "slp_vectorizer.h"
#include "program.h"
#include "loop_vectorizer.h"
#include "optimizer.h"
#include "adapter_preprocessor.h"
#include "vector_splitter.h"
#include "desugaring.h"

TLANG_NAMESPACE_BEGIN

class TikzGen : public Visitor {
 public:
  std::string graph;
  TikzGen() : Visitor(Visitor::Order::parent_first) {
  }

  std::string expr_name(Expr expr) {
    std::string members = "";
    if (!expr) {
      TC_ERROR("expr = 0");
    }
    if (expr->members.size()) {
      members = "[";
      bool first = true;
      for (auto m : expr->members) {
        if (!first)
          members += ", ";
        members += fmt::format("{}", m->id);
        first = false;
      }
      members += "]";
    }
    return fmt::format("\"({}){}{}\"", expr->id, members,
                       expr->node_type_name());
  }

  void link(Expr a, Expr b) {
    graph += fmt::format("{} -> {}; ", expr_name(a), expr_name(b));
  }

  void visit(Expr &expr) override {
    for (auto &ch : expr->ch) {
      link(expr, ch);
    }
  }
};

void visualize_IR(std::string fn, Expr &expr) {
  TikzGen gen;
  expr.accept(gen);
  auto cmd =
      fmt::format("python3 {}/projects/taichi_lang/make_graph.py {} '{}'",
                  get_repo_dir(), fn, gen.graph);
  trash(system(cmd.c_str()));
}

#if (0)
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
#endif

void CPUCodeGen::codegen(Kernel &kernel) {
  // TC_ASSERT(mode == Mode::vector);
  this->prog = &kernel.program;
  this->current_kernel = &kernel;
  this->simd_width = prog->config.simd_width;
  this->num_groups = kernel.parallel_instances;

  auto snode = prog->current_snode;
  while (snode->type == SNodeType::forked) {
    snode = snode->parent;
  }
  has_residual = kernel.parallel_instances > 1 &&
                 (snode->type == SNodeType::indirect ||
                  snode->parent->type == SNodeType::dynamic);

  {
    CODE_REGION(header);
    generate_header();
  }

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

  {
    CODE_REGION(tail);
    code_suffix = "";
    generate_tail();
  }
}

FunctionType CPUCodeGen::get(Program &prog, Kernel &kernel) {
  // auto mode = CPUCodeGen::Mode::vv;
  auto mode = CPUCodeGen::Mode::intrinsics;
  auto simd_width = prog.config.simd_width;
  this->mode = mode;
  this->simd_width = simd_width;
  codegen(kernel);
  return compile();
}

FunctionType Program::compile(Kernel &kernel) {
  FunctionType ret = nullptr;
  if (config.arch == Arch::x86_64) {
    CPUCodeGen codegen;
    if (!kernel.name.empty()) {
      codegen.source_name = kernel.name + ".cpp";
    }
    ret = codegen.get(*this, kernel);
  } else if (config.arch == Arch::gpu) {
    TC_NOT_IMPLEMENTED
    // GPUCodeGen codegen;
    // function = codegen.get(*this);
  } else {
    TC_NOT_IMPLEMENTED;
  }
  TC_ASSERT(ret);
  return ret;
}

std::string CodeGenBase::get_source_fn() {
  return fmt::format("{}/{}/{}", get_project_fn(), folder, source_name);
}

FunctionType CPUCodeGen::compile() {
  write_code_to_file();
  auto cmd = get_current_program().config.compile_cmd(get_source_fn(),
                                                      get_library_fn());
  auto compile_ret = std::system(cmd.c_str());
  if (compile_ret != 0) {
    auto cmd = get_current_program().config.compile_cmd(get_source_fn(),
                                                        get_library_fn(), true);
    trash(std::system(cmd.c_str()));
    TC_ERROR("Source {} compilation failed.", get_source_fn());
  }
  disassemble();
  return load_function();
}

TLANG_NAMESPACE_END
