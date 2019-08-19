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

Kernel::Kernel(Program &program,
               std::function<void()> func,
               std::string name,
               bool grad)
    : program(program), name(name), grad(grad) {
  is_reduction = false;
  compiled = nullptr;
  benchmarking = false;
  taichi::Tlang::context = std::make_unique<FrontendContext>();
  ir_holder = taichi::Tlang::context->get_root();
  ir = ir_holder.get();

  program.current_kernel = this;
  program.start_function_definition(this);
  func();
  program.end_function_definition();
  program.current_kernel = nullptr;

  if (!program.config.lazy_compilation)
    compile();
}

void Kernel::compile() {
  program.current_kernel = this;
  compiled = program.compile(*this);
  program.current_kernel = nullptr;
}

void Kernel::operator()() {
  if (!compiled)
    compile();
  std::vector<void *> host_buffers(args.size());
  std::vector<void *> device_buffers(args.size());
  if (program.config.arch == Arch::gpu) {
    // copy data to GRAM
    bool has_buffer = false;
    for (int i = 0; i < (int)args.size(); i++) {
      if (args[i].is_nparray) {
        has_buffer = true;
        cudaMalloc(&device_buffers[i], args[i].size);
        // replace host buffer with device buffer
        host_buffers[i] = program.context.get_arg<void *>(i);
        set_arg_nparray(i, (uint64)device_buffers[i], args[i].size);
        cudaMemcpy(device_buffers[i], host_buffers[i], args[i].size,
                   cudaMemcpyHostToDevice);
      }
    }
    if (has_buffer)
      cudaDeviceSynchronize();
    auto c = program.get_context();
    compiled(c);
    if (has_buffer)
      cudaDeviceSynchronize();
    for (int i = 0; i < (int)args.size(); i++) {
      if (args[i].is_nparray) {
        cudaMemcpy(host_buffers[i], device_buffers[i], args[i].size,
                   cudaMemcpyDeviceToHost);
        cudaFree(device_buffers[i]);
      }
    }
  } else {
    auto c = program.get_context();
    compiled(c);
  }
  program.sync = false;
}

void Kernel::set_arg_float(int i, float64 d) {
  TC_ASSERT_INFO(args[i].is_nparray == false,
                 "Setting scalar value to numpy array argument is not allowed");
  auto dt = args[i].dt;
  if (dt == DataType::f32) {
    program.context.set_arg(i, (float32)d);
  } else if (dt == DataType::f64) {
    program.context.set_arg(i, (float64)d);
  } else if (dt == DataType::i32) {
    program.context.set_arg(i, (int32)d);
  } else if (dt == DataType::i64) {
    program.context.set_arg(i, (int64)d);
  } else if (dt == DataType::i16) {
    program.context.set_arg(i, (int16)d);
  } else if (dt == DataType::u16) {
    program.context.set_arg(i, (uint16)d);
  } else if (dt == DataType::u32) {
    program.context.set_arg(i, (uint32)d);
  } else if (dt == DataType::u64) {
    program.context.set_arg(i, (uint64)d);
  } else {
    TC_NOT_IMPLEMENTED
  }
}

void Kernel::set_arg_int(int i, int64 d) {
  TC_ASSERT_INFO(args[i].is_nparray == false,
                 "Setting scalar value to numpy array argument is not allowed");
  auto dt = args[i].dt;
  if (dt == DataType::i32) {
    program.context.set_arg(i, (int32)d);
  } else if (dt == DataType::i64) {
    program.context.set_arg(i, (int64)d);
  } else if (dt == DataType::i16) {
    program.context.set_arg(i, (int16)d);
  } else if (dt == DataType::u16) {
    program.context.set_arg(i, (uint16)d);
  } else if (dt == DataType::u32) {
    program.context.set_arg(i, (uint32)d);
  } else if (dt == DataType::u64) {
    program.context.set_arg(i, (uint64)d);
  } else if (dt == DataType::f32) {
    program.context.set_arg(i, (float32)d);
  } else if (dt == DataType::f64) {
    program.context.set_arg(i, (float64)d);
  } else {
    TC_NOT_IMPLEMENTED
  }
}

void Kernel::set_arg_nparray(int i, uint64 d, uint64 size) {
  TC_ASSERT_INFO(args[i].is_nparray,
                 "Setting numpy array to scalar argument is not allowed");
  args[i].size = size;
  program.context.set_arg(i, d);
}

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

TLANG_NAMESPACE_END
