// Program, which is a context for a taichi program execution

#include <taichi/common/task.h>
#include <taichi/taichi>
#include "program.h"
#include "snode.h"
#include "backends/struct.h"
#include "backends/cpu.h"
#include "backends/gpu.h"
#if defined(CUDA_FOUND)
#include <cuda_runtime.h>
#endif

TLANG_NAMESPACE_BEGIN

Program *current_program = nullptr;
SNode root;

FunctionType Program::compile(Kernel &kernel) {
  FunctionType ret = nullptr;
  if (config.arch == Arch::x86_64) {
    CPUCodeGen codegen(kernel.name);
    ret = codegen.compile(*this, kernel);
  } else if (config.arch == Arch::gpu) {
    GPUCodeGen codegen(kernel.name);
    ret = codegen.compile(*this, kernel);
  } else {
    TC_NOT_IMPLEMENTED;
  }
  TC_ASSERT(ret);
  return ret;
}

void Program::materialize_layout() {
  std::unique_ptr<StructCompiler> scomp = StructCompiler::make(config.use_llvm);
  scomp->run(root);
  layout_fn = scomp->get_source_path();
  data_structure = scomp->creator();
  profiler_print_gpu = scomp->profiler_print;
  profiler_clear_gpu = scomp->profiler_clear;
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
#endif
  llvm_context_host = std::make_unique<TaichiLLVMContext>(Arch::x86_64);
  // llvm_context_device is initialized before kernel compilation
  UnifiedAllocator::create();
  TC_ASSERT(current_program == nullptr);
  current_program = this;
  config = default_compile_config;
  config.arch = arch;
  current_kernel = nullptr;
  snode_root = nullptr;
  index_counter = 0;
  sync = true;
  llvm_runtime = nullptr;
}

void Program::initialize_device_llvm_context() {
  if (config.arch == Arch::gpu && config.use_llvm) {
    if (llvm_context_device == nullptr)
      llvm_context_device = std::make_unique<TaichiLLVMContext>(Arch::gpu);
  }
}

TLANG_NAMESPACE_END
