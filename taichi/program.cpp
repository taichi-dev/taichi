// Program, which is a context for a taichi program execution

#include "program.h"

#include <taichi/common/task.h>
#include <taichi/platform/metal/metal_api.h>

#include "backends/codegen_cuda.h"
#include "backends/codegen_metal.h"
#include "backends/codegen_x86.h"
#include "backends/struct.h"
#include "backends/struct_metal.h"
#include "unified_allocator.h"
#include "snode.h"

#if defined(TI_WITH_CUDA)

#include <cuda_runtime.h>

#include "backends/cuda_context.h"

#endif

TLANG_NAMESPACE_BEGIN

void assert_failed_host(const char *msg) {
  TI_ERROR("Assertion failure: {}", msg);
}

void *taichi_allocate_aligned(Program *prog,
                              std::size_t size,
                              std::size_t alignment) {
  return prog->memory_pool->allocate(size, alignment);
}

Program *current_program = nullptr;
std::atomic<int> Program::num_instances;

Program::Program(Arch arch) {
#if !defined(TI_WITH_CUDA)
  if (arch == Arch::cuda) {
    TI_WARN("Taichi is not compiled with CUDA.");
    TI_WARN("Falling back to x86_64");
    arch = Arch::x64;
  }
#else
  if (!cuda_context) {
    cuda_context = std::make_unique<CUDAContext>();
    if (!cuda_context->detected()) {
      TI_WARN("No CUDA device detected.");
      TI_WARN("Falling back to x86_64");
      arch = Arch::x64;
    }
  }
#endif
  if (arch == Arch::metal) {
    if (!metal::is_metal_api_available()) {
      TI_WARN("No Metal API detected, falling back to x86_64");
      arch = Arch::x64;
    }
  }
  memory_pool = std::make_unique<MemoryPool>(this);
  TI_ASSERT_INFO(num_instances == 0, "Only one instance at a time");
  total_compilation_time = 0;
  num_instances += 1;
  SNode::counter = 0;
  // llvm_context_device is initialized before kernel compilation
  TI_ASSERT(current_program == nullptr);
  current_program = this;
  config = default_compile_config;
  config.arch = arch;
  if (config.use_llvm) {
    llvm_context_host = std::make_unique<TaichiLLVMContext>(Arch::x64);
    profiler = make_profiler(arch);
  }
  current_kernel = nullptr;
  sync = true;
  llvm_runtime = nullptr;
  finalized = false;
  snode_root = std::make_unique<SNode>(0, SNodeType::root);

  if (config.debug) {
    TI_DEBUG("Program arch={}", arch_name(arch));
  }
}

FunctionType Program::compile(Kernel &kernel) {
  auto start_t = Time::get_time();
  TI_AUTO_PROF;
  FunctionType ret = nullptr;
  if (kernel.arch == Arch::x64) {
    CPUCodeGen codegen(kernel.name);
    ret = codegen.compile(*this, kernel);
  } else if (kernel.arch == Arch::cuda) {
    GPUCodeGen codegen(kernel.name);
    ret = codegen.compile(*this, kernel);
  } else if (kernel.arch == Arch::metal) {
    metal::MetalCodeGen codegen(kernel.name, &metal_struct_compiled_.value());
    ret = codegen.compile(*this, kernel, metal_runtime_.get());
  } else {
    TI_NOT_IMPLEMENTED;
  }
  TI_ASSERT(ret);
  total_compilation_time += Time::get_time() - start_t;
  return ret;
}

// For CPU and CUDA archs only
void Program::initialize_runtime_system(StructCompiler *scomp) {
  auto tlctx = llvm_context_host.get();
  auto runtime = tlctx->runtime_jit_module;

  auto initialize_runtime2 =
      tlctx->lookup_function<void(void *, int, int)>("Runtime_initialize2");

  auto set_assert_failed =
      tlctx->lookup_function<void(void *, void *)>("Runtime_set_assert_failed");

  auto allocate_ambient =
      tlctx->lookup_function<void(void *, int)>("Runtime_allocate_ambient");

  auto initialize_allocator =
      tlctx->lookup_function<void *(void *, int, std::size_t)>(
          "NodeAllocator_initialize");

  auto runtime_initialize_thread_pool =
      tlctx->lookup_function<void(void *, void *, void *)>(
          "Runtime_initialize_thread_pool");

  // By the time this creator is called, "this" is already destroyed.
  // Therefore it is necessary to capture members by values.
  auto snodes = scomp->snodes;
  auto root_id = snode_root->id;

  TI_TRACE("Allocating data structure of size {} B", scomp->root_size);

  runtime->call<void *, void *, std::size_t, void *, void *>(
      "Runtime_initialize", &llvm_runtime, this, (std::size_t)scomp->root_size,
      (void *)&taichi_allocate_aligned, (void *)std::printf);
  TI_TRACE("Runtime initialized");

  auto mem_req_queue = tlctx->lookup_function<void *(void *)>(
      "Runtime_get_mem_req_queue")(llvm_runtime);
  memory_pool->set_queue((MemRequestQueue *)mem_req_queue);

  initialize_runtime2(llvm_runtime, root_id, (int)snodes.size());
  for (int i = 0; i < (int)snodes.size(); i++) {
    if (snodes[i]->type == SNodeType::pointer ||
        snodes[i]->type == SNodeType::dynamic) {
      std::size_t node_size;
      if (snodes[i]->type == SNodeType::pointer)
        node_size = tlctx->get_type_size(
            scomp->snode_attr[snodes[i]].llvm_element_type);
      else {
        // dynamic. Allocators are for the chunks
        node_size = sizeof(void *) +
                    tlctx->get_type_size(
                        scomp->snode_attr[snodes[i]].llvm_element_type) *
                        snodes[i]->chunk_size;
      }
      TI_TRACE("Initializing allocator for snode {} (node size {})",
               snodes[i]->id, node_size);
      auto rt = llvm_runtime;
      initialize_allocator(rt, i, node_size);
      TI_TRACE("Allocating ambient element for snode {} (node size {})",
               snodes[i]->id, node_size);
      allocate_ambient(rt, i);
    }
  }

  runtime_initialize_thread_pool(llvm_runtime, &thread_pool,
                                 (void *)ThreadPool::static_run);
  set_assert_failed(llvm_runtime, (void *)assert_failed_host);

  if (arch_use_host_memory(config.arch)) {
    // Profiler functions can only be called on host kernels
    tlctx->lookup_function<void(void *, void *)>("Runtime_set_profiler")(
        llvm_runtime, profiler.get());

    tlctx->lookup_function<void(void *, void *)>("Runtime_set_profiler_start")(
        llvm_runtime, (void *)&ProfilerBase::profiler_start);
    tlctx->lookup_function<void(void *, void *)>("Runtime_set_profiler_stop")(
        llvm_runtime, (void *)&ProfilerBase::profiler_stop);
  }
}

void Program::materialize_layout() {
  // always use arch=x86_64 since this is for host accessors
  // TODO: arch may also be arm etc.
  std::unique_ptr<StructCompiler> scomp = StructCompiler::make(this, Arch::x64);
  scomp->run(*snode_root, true);

  if (arch_is_cpu(config.arch) || config.arch == Arch::cuda ||
      config.arch == Arch::metal) {
    initialize_runtime_system(scomp.get());
  }

  if (config.arch == Arch::cuda && config.use_llvm) {
    initialize_device_llvm_context();
    // llvm_context_device->get_init_module();
    std::unique_ptr<StructCompiler> scomp_gpu =
        StructCompiler::make(this, Arch::cuda);
    scomp_gpu->run(*snode_root, false);
  } else if (config.arch == Arch::metal) {
    TI_ASSERT_INFO(config.use_llvm,
                   "Metal arch requires that LLVM being enabled");
    metal::MetalStructCompiler scomp;
    metal_struct_compiled_ = scomp.run(*snode_root);
    if (metal_runtime_ == nullptr) {
      metal::MetalRuntime::Params params;
      params.root_size = metal_struct_compiled_->root_size;
      params.llvm_runtime = llvm_runtime;
      params.llvm_ctx = get_llvm_context(get_host_arch());
      params.config = &config;
      params.mem_pool = memory_pool.get();
      params.profiler = profiler.get();
      metal_runtime_ = std::make_unique<metal::MetalRuntime>(std::move(params));
    }
    TI_INFO("Metal root buffer size: {} B", metal_struct_compiled_->root_size);
  }
}

void Program::synchronize() {
  if (!sync) {
    if (config.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
      cudaDeviceSynchronize();
#else
      TI_ERROR("No CUDA support");
#endif
    } else if (config.arch == Arch::metal) {
      metal_runtime_->synchronize();
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
    TI_ASSERT(v % 1024 == 0);
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
    TI_ASSERT(ofs);
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

    visit(snode_root.get());

    auto tail = R"(
\end{tikzpicture}
\end{document}
)";
    emit(tail);
  }
  trash(system(fmt::format("pdflatex {}", fn).c_str()));
}

void Program::initialize_device_llvm_context() {
  if (config.arch == Arch::cuda) {
    if (llvm_context_device == nullptr)
      llvm_context_device = std::make_unique<TaichiLLVMContext>(Arch::cuda);
  }
}

Kernel &Program::get_snode_reader(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
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
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_arg(DataType::i32, false);
  auto ret_val = ker.insert_arg(snode->dt, false);
  ker.mark_arg_return_value(ret_val);
  return ker;
}

Kernel &Program::get_snode_writer(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
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
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_arg(DataType::i32, false);
  ker.insert_arg(snode->dt, false);
  return ker;
}

void Program::finalize() {
  synchronize();
  current_program = nullptr;
  for (auto &dll : loaded_dlls) {
#if defined(TI_PLATFORM_UNIX)
    dlclose(dll);
#else
    TI_NOT_IMPLEMENTED
#endif
  }
  memory_pool->terminate();
  finalized = true;
  num_instances -= 1;
}

Program::~Program() {
  if (!finalized)
    finalize();
}

TLANG_NAMESPACE_END
