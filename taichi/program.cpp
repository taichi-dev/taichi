// Program, which is a context for a taichi program execution

#include <taichi/common/task.h>
#include "program.h"
#include "snode.h"
#include "backends/struct.h"
#include "backends/codegen_x86.h"
#include "backends/codegen_cuda.h"

#if defined(CUDA_FOUND)

#include <cuda_runtime.h>
#include "backends/cuda_context.h"

#endif

TLANG_NAMESPACE_BEGIN

Program *current_program = nullptr;
std::atomic<int> Program::num_instances;
SNode root;

FunctionType Program::compile(Kernel &kernel) {
  auto start_t = Time::get_time();
  TI_AUTO_PROF;
  FunctionType ret = nullptr;
  if (kernel.arch == Arch::x86_64) {
    CPUCodeGen codegen(kernel.name);
    ret = codegen.compile(*this, kernel);
  } else if (kernel.arch == Arch::gpu) {
    GPUCodeGen codegen(kernel.name);
    ret = codegen.compile(*this, kernel);
  } else {
    TC_NOT_IMPLEMENTED;
  }
  TC_ASSERT(ret);
  total_compilation_time += Time::get_time() - start_t;
  return ret;
}

void Program::materialize_layout() {
  // always use arch=x86_64 since this is for host accessors
  std::unique_ptr<StructCompiler> scomp =
      StructCompiler::make(config.use_llvm, Arch::x86_64);
  scomp->run(root, true);
  layout_fn = scomp->get_source_path();
  data_structure = scomp->creator();
  profiler_print_gpu = scomp->profiler_print;
  profiler_clear_gpu = scomp->profiler_clear;

  if (config.arch == Arch::gpu && config.use_llvm) {
    initialize_device_llvm_context();
    // llvm_context_device->get_init_module();
    std::unique_ptr<StructCompiler> scomp_gpu =
        StructCompiler::make(config.use_llvm, Arch::gpu);
    scomp_gpu->run(root, false);
  }
}

void Program::synchronize() {
  if (!sync) {
    if (config.arch == Arch::gpu) {
#if defined(CUDA_FOUND)
      cudaDeviceSynchronize();
#else
      TC_ERROR("No CUDA support");
#endif
    }
    sync = true;
  }
}

std::string capitalize_first(std::string s) {
  s[0] = std::toupper(s[0]);
  return s;
}

std::string latex_short_digit(int v) {
  std::string units = "KMGT";
  int unit_id = -1;
  while (v >= 1024 && unit_id + 1 < (int)units.size()) {
    TC_ASSERT(v % 1024 == 0);
    v /= 1024;
    unit_id++;
  }
  if (unit_id != -1)
    return fmt::format("{}\\mathrm{{{}}}", v, units[unit_id]);
  else
    return std::to_string(v);
}

void Program::visualize_layout(const std::string &fn) {
  {
    std::ofstream ofs(fn);
    TC_ASSERT(ofs);
    auto emit = [&](std::string str) { ofs << str; };

    auto header = R"(
\documentclass[tikz, border=16pt]{standalone}
\usepackage{latexsym}
\usepackage{tikz-qtree,tikz-qtree-compat,ulem}
\begin{document}
\begin{tikzpicture}[level distance=40pt]
\tikzset{level 1/.style={sibling distance=-5pt}}
  \tikzset{edge from parent/.style={draw,->,
    edge from parent path={(\tikzparentnode.south) -- +(0,-4pt) -| (\tikzchildnode)}}}
  \tikzset{every tree node/.style={align=center, font=\small}}
\Tree)";
    emit(header);

    std::function<void(SNode * snode)> visit = [&](SNode *snode) {
      emit("[.{");
      if (snode->type == SNodeType::place) {
        emit(snode->name);
      } else {
        emit("\\textbf{" + capitalize_first(snode_type_name(snode->type)) +
             "}");
      }

      std::string indices;
      for (int i = 0; i < max_num_indices; i++) {
        if (snode->extractors[i].active) {
          int nb = snode->extractors[i].num_bits;
          int start = snode->extractors[i].start + nb;
          indices += fmt::format(
              R"($\mathbf{{{}}}^{{\mathbf{{{}b}}:{}}}_{{\mathbf{{{}b}}:{}}}$)",
              std::string(1, 'I' + i), start, latex_short_digit(1 << start), nb,
              latex_short_digit(1 << nb));
        }
      }
      if (!indices.empty())
        emit("\\\\" + indices);
      if (snode->type == SNodeType::place) {
        emit("\\\\" + data_type_short_name(snode->dt));
      }
      emit("} ");

      for (int i = 0; i < (int)snode->ch.size(); i++) {
        visit(snode->ch[i].get());
      }
      emit("]");
    };

    visit(snode_root);

    auto tail = R"(
\end{tikzpicture}
\end{document}
)";
    emit(tail);
  }
  trash(system(fmt::format("pdflatex {}", fn).c_str()));
}

Program::Program(Arch arch) {
#if !defined(CUDA_FOUND)
  if (arch == Arch::gpu) {
    TC_WARN("CUDA not found. GPU is not supported.");
    TC_WARN("Falling back to x86_64");
    arch = Arch::x86_64;
  }
#else
  if (!cuda_context) {
    cuda_context = std::make_unique<CUDAContext>();
  }
#endif
  TC_ASSERT_INFO(num_instances == 0, "Only one instance at a time");
  total_compilation_time = 0;
  num_instances += 1;
  SNode::counter = 0;
  // llvm_context_device is initialized before kernel compilation
  UnifiedAllocator::create();
  TC_ASSERT(current_program == nullptr);
  current_program = this;
  config = default_compile_config;
  config.arch = arch;
  if (config.use_llvm) {
    llvm_context_host = std::make_unique<TaichiLLVMContext>(Arch::x86_64);
    if (config.arch == Arch::x86_64) {
      profiler_llvm = std::make_unique<CPUProfiler>();
    } else {
      profiler_llvm = std::make_unique<GPUProfiler>();
    }
  }
  auto env_debug = getenv("TI_DEBUG");
  if (env_debug && env_debug == std::string("1"))
    config.debug = true;
  current_kernel = nullptr;
  snode_root = nullptr;
  sync = true;
  llvm_runtime = nullptr;
  finalized = false;
}

void Program::initialize_device_llvm_context() {
  if (config.arch == Arch::gpu && config.use_llvm) {
    if (llvm_context_device == nullptr)
      llvm_context_device = std::make_unique<TaichiLLVMContext>(Arch::gpu);
  }
}

Kernel &Program::get_snode_reader(SNode *snode) {
  TC_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_reader_{}", snode->id);
  auto &ker = kernel([&] {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      indices.push_back(Expr::make<ArgLoadExpression>(i));
    }
    auto ret = Stmt::make<FrontendArgStoreStmt>(
        snode->num_active_indices, load_if_ptr((snode->expr)[indices]));
    current_ast_builder().insert(std::move(ret));
  });
  ker.set_arch(get_host_arch());
  ker.name = kernel_name;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_arg(DataType::i32, false);
  auto ret_val = ker.insert_arg(snode->dt, false);
  ker.mark_arg_return_value(ret_val);
  return ker;
}

Kernel &Program::get_snode_writer(SNode *snode) {
  TC_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_writer_{}", snode->id);
  auto &ker = kernel([&] {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      indices.push_back(Expr::make<ArgLoadExpression>(i));
    }
    (snode->expr)[indices] =
        Expr::make<ArgLoadExpression>(snode->num_active_indices);
  });
  ker.set_arch(get_host_arch());
  ker.name = kernel_name;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_arg(DataType::i32, false);
  ker.insert_arg(snode->dt, false);
  return ker;
}

TLANG_NAMESPACE_END
