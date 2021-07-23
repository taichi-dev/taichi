// Program, context for Taichi program execution

#include "program.h"

#include "taichi/ir/statements.h"
#include "taichi/program/extension.h"
#include "taichi/backends/metal/api.h"
#include "taichi/backends/opengl/opengl_api.h"
#if defined(TI_WITH_CUDA)
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/codegen_cuda.h"
#include "taichi/backends/cuda/cuda_context.h"
#endif
#include "taichi/backends/metal/codegen_metal.h"
#include "taichi/backends/opengl/codegen_opengl.h"
#include "taichi/backends/cpu/codegen_cpu.h"
#include "taichi/struct/struct.h"
#include "taichi/struct/struct_llvm.h"
#include "taichi/backends/metal/aot_module_builder_impl.h"
#include "taichi/backends/metal/struct_metal.h"
#include "taichi/backends/wasm/aot_module_builder_impl.h"
#include "taichi/backends/opengl/struct_opengl.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/system/unified_allocator.h"
#include "taichi/system/timeline.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/program/async_engine.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/util/statistics.h"
#include "taichi/util/str.h"
#if defined(TI_WITH_CC)
#include "taichi/backends/cc/struct_cc.h"
#include "taichi/backends/cc/cc_layout.h"
#include "taichi/backends/cc/codegen_cc.h"
#endif

#if defined(TI_ARCH_x64)
// For _MM_SET_FLUSH_ZERO_MODE
#include <xmmintrin.h>
#endif

namespace taichi {
namespace lang {
namespace {

void assert_failed_host(const char *msg) {
  TI_ERROR("Assertion failure: {}", msg);
}

void *taichi_allocate_aligned(Program *prog,
                              std::size_t size,
                              std::size_t alignment) {
  return prog->memory_pool->allocate(size, alignment);
}

inline uint64 *allocate_result_buffer_default(Program *prog) {
  return (uint64 *)taichi_allocate_aligned(
      prog, sizeof(uint64) * taichi_result_buffer_entries, 8);
}

bool is_cuda_no_unified_memory(const CompileConfig &config) {
  return (config.arch == Arch::cuda && !config.use_unified_memory);
}

}  // namespace

Program *current_program = nullptr;
std::atomic<int> Program::num_instances;

Program::Program(Arch desired_arch) : snode_rw_accessors_bank_(this) {
  TI_TRACE("Program initializing...");

  // For performance considerations and correctness of CustomFloatType
  // operations, we force floating-point operations to flush to zero on all
  // backends (including CPUs).
#if defined(TI_ARCH_x64)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#else
  // Enforce flush to zero on arm64 CPUs
  // https://developer.arm.com/documentation/100403/0201/register-descriptions/advanced-simd-and-floating-point-registers/aarch64-register-descriptions/fpcr--floating-point-control-register?lang=en
  std::uint64_t fpcr;
  __asm__ __volatile__("");
  __asm__ __volatile__("MRS %0, FPCR" : "=r"(fpcr));
  __asm__ __volatile__("");
  __asm__ __volatile__("MSR FPCR, %0"
                       :
                       : "ri"(fpcr | (1 << 24)));  // Bit 24 is FZ
  __asm__ __volatile__("");
#endif

  auto arch = desired_arch;
  if (arch == Arch::cuda) {
    runtime_mem_info = Runtime::create(arch);
    if (!runtime_mem_info) {
      TI_WARN("Taichi is not compiled with CUDA.");
      arch = host_arch();
    } else if (!is_cuda_api_available()) {
      TI_WARN("No CUDA driver API detected.");
      arch = host_arch();
    } else if (!runtime_mem_info->detected()) {
      TI_WARN("No CUDA device detected.");
      arch = host_arch();
    } else {
      // CUDA runtime created successfully
    }
    if (arch != Arch::cuda) {
      TI_WARN("Falling back to {}.", arch_name(host_arch()));
    }
  }
  if (arch == Arch::metal) {
    if (!metal::is_metal_api_available()) {
      TI_WARN("No Metal API detected.");
      arch = host_arch();
    }
  }
  if (arch == Arch::opengl) {
    if (!opengl::is_opengl_api_available()) {
      TI_WARN("No OpenGL API detected.");
      arch = host_arch();
    }
  }

  if (arch == Arch::cc) {
#ifdef TI_WITH_CC
    cc_program = std::make_unique<cccp::CCProgram>(this);
#else
    TI_WARN("No C backend detected.");
    arch = host_arch();
#endif
  }

  if (arch != desired_arch) {
    TI_WARN("Falling back to {}", arch_name(arch));
  }

  memory_pool = std::make_unique<MemoryPool>(this);
  TI_ASSERT_INFO(num_instances == 0, "Only one instance at a time");
  total_compilation_time = 0;
  num_instances += 1;
  SNode::counter = 0;
  // |llvm_context_device| is initialized before kernel compilation
  TI_ASSERT(current_program == nullptr);
  current_program = this;
  config = default_compile_config;
  config.arch = arch;

  thread_pool = std::make_unique<ThreadPool>(config.cpu_max_num_threads);

  llvm_context_host = std::make_unique<TaichiLLVMContext>(host_arch());
  llvm_context_host->init_runtime_jit_module();
  // TODO: Can we initialize |llvm_context_device| here?
  profiler = make_profiler(arch);

  preallocated_device_buffer = nullptr;

  if (config.kernel_profiler && runtime_mem_info) {
    runtime_mem_info->set_profiler(profiler.get());
  }
#if defined(TI_WITH_CUDA)
  if (config.arch == Arch::cuda) {
    if (config.kernel_profiler) {
      CUDAContext::get_instance().set_profiler(profiler.get());
    } else {
      CUDAContext::get_instance().set_profiler(nullptr);
    }
  }
#endif

  result_buffer = nullptr;
  current_callable = nullptr;
  sync = true;
  llvm_runtime = nullptr;
  finalized = false;

  if (config.async_mode) {
    TI_WARN("Running in async mode. This is experimental.");
    TI_ASSERT(is_extension_supported(config.arch, Extension::async_mode));
    async_engine = std::make_unique<AsyncEngine>(
        this, [this](Kernel &kernel, OffloadedStmt *offloaded) {
          return this->compile_to_backend_executable(kernel, offloaded);
        });
  }

  // TODO: allow users to run in debug mode without out-of-bound checks
  if (config.debug)
    config.check_out_of_bound = true;

  if (!is_extension_supported(config.arch, Extension::assertion)) {
    if (config.check_out_of_bound) {
      TI_WARN("Out-of-bound access checking is not supported on arch={}",
              arch_name(config.arch));
      config.check_out_of_bound = false;
    }
  }

  if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    int num_SMs;
    CUDADriver::get_instance().device_get_attribute(
        &num_SMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, nullptr);
    int query_max_block_dim;
    CUDADriver::get_instance().device_get_attribute(
        &query_max_block_dim, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, nullptr);

    if (config.max_block_dim == 0) {
      config.max_block_dim = query_max_block_dim;
    }

    if (config.saturating_grid_dim == 0) {
      // each SM can have 16-32 resident blocks
      config.saturating_grid_dim = num_SMs * 32;
    }
#endif
  }

  if (arch_is_cpu(arch)) {
    config.max_block_dim = 1024;
  }

  stat.clear();

  Timelines::get_instance().set_enabled(config.timeline);

  TI_TRACE("Program ({}) arch={} initialized.", fmt::ptr(this),
           arch_name(arch));
}

TypeFactory &Program::get_type_factory() {
  TI_WARN(
      "Program::get_type_factory() will be deprecated, Please use "
      "TypeFactory::get_instance()");
  return TypeFactory::get_instance();
}

Function *Program::create_function(const FunctionKey &func_key) {
  TI_TRACE("Creating function {}...", func_key.get_full_name());
  functions.emplace_back(std::make_unique<Function>(this, func_key));
  TI_ASSERT(function_map.count(func_key) == 0);
  function_map[func_key] = functions.back().get();
  return functions.back().get();
}

FunctionType Program::compile(Kernel &kernel) {
  auto start_t = Time::get_time();
  TI_AUTO_PROF;
  FunctionType ret = nullptr;
  if (Kernel::supports_lowering(kernel.arch)) {
    kernel.lower();
    ret = compile_to_backend_executable(kernel, /*offloaded=*/nullptr);
  } else if (kernel.arch == Arch::opengl) {
    opengl::OpenglCodeGen codegen(kernel.name, &opengl_struct_compiled_.value(),
                                  opengl_kernel_launcher_.get());
    ret = codegen.compile(*this, kernel);
#ifdef TI_WITH_CC
  } else if (kernel.arch == Arch::cc) {
    ret = cccp::compile_kernel(&kernel);
#endif
  } else {
    TI_NOT_IMPLEMENTED;
  }
  TI_ASSERT(ret);
  total_compilation_time += Time::get_time() - start_t;
  return ret;
}

FunctionType Program::compile_to_backend_executable(Kernel &kernel,
                                                    OffloadedStmt *offloaded) {
  if (arch_is_cpu(kernel.arch) || kernel.arch == Arch::cuda) {
    auto codegen = KernelCodeGen::create(kernel.arch, &kernel, offloaded);
    return codegen->compile();
  } else if (kernel.arch == Arch::metal) {
    return metal::compile_to_metal_executable(&kernel, metal_kernel_mgr_.get(),
                                              &metal_compiled_structs_.value(),
                                              offloaded);
  }
  TI_NOT_IMPLEMENTED;
  return nullptr;
}

void Program::materialize_runtime() {
  if (arch_uses_llvm(config.arch)) {
    initialize_llvm_runtime_system();
  }
}

// For CPU and CUDA archs only
void Program::initialize_llvm_runtime_system() {
  maybe_initialize_cuda_llvm_context();

  std::size_t prealloc_size = 0;
  TaichiLLVMContext *tlctx = nullptr;
  if (is_cuda_no_unified_memory(config)) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().malloc(
        (void **)&result_buffer, sizeof(uint64) * taichi_result_buffer_entries);
    const auto total_mem = runtime_mem_info->get_total_memory();
    if (config.device_memory_fraction == 0) {
      TI_ASSERT(config.device_memory_GB > 0);
      prealloc_size = std::size_t(config.device_memory_GB * (1UL << 30));
    } else {
      prealloc_size = std::size_t(config.device_memory_fraction * total_mem);
    }
    TI_ASSERT(prealloc_size <= total_mem);

    TI_TRACE("Allocating device memory {:.2f} GB",
             1.0 * prealloc_size / (1UL << 30));

    CUDADriver::get_instance().malloc(&preallocated_device_buffer,
                                      prealloc_size);
    CUDADriver::get_instance().memset(preallocated_device_buffer, 0,
                                      prealloc_size);
    tlctx = llvm_context_device.get();
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    result_buffer = allocate_result_buffer_default(this);
    tlctx = llvm_context_host.get();
  }
  auto *const runtime_jit = tlctx->runtime_jit_module;

  // Starting random state for the program calculated using the random seed.
  // The seed is multiplied by 2^20 so that two programs with different seeds
  // will not have overlapping random states in any thread.
  int starting_rand_state = config.random_seed * 1048576;

  // Number of random states. One per CPU/CUDA thread.
  int num_rand_states = 0;

  if (config.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    // It is important to make sure that every CUDA thread has its own random
    // state so that we do not need expensive per-state locks.
    num_rand_states = config.saturating_grid_dim * config.max_block_dim;
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    num_rand_states = config.cpu_max_num_threads;
  }

  TI_TRACE("Allocating {} random states (used by CUDA only)", num_rand_states);

  runtime_jit->call<void *, void *, std::size_t, void *, int, int, void *,
                    void *, void *>(
      "runtime_initialize", result_buffer, this, prealloc_size,
      preallocated_device_buffer, starting_rand_state, num_rand_states,
      (void *)&taichi_allocate_aligned, (void *)std::printf,
      (void *)std::vsnprintf);

  TI_TRACE("LLVMRuntime initialized (excluding `root`)");
  llvm_runtime = fetch_result<void *>(taichi_result_buffer_ret_value_id);
  TI_TRACE("LLVMRuntime pointer fetched");

  if (arch_use_host_memory(config.arch) || config.use_unified_memory) {
    runtime_jit->call<void *>("runtime_get_mem_req_queue", llvm_runtime);
    auto mem_req_queue =
        fetch_result<void *>(taichi_result_buffer_ret_value_id);
    memory_pool->set_queue((MemRequestQueue *)mem_req_queue);
  }

  if (arch_use_host_memory(config.arch)) {
    runtime_jit->call<void *, void *, void *>(
        "LLVMRuntime_initialize_thread_pool", llvm_runtime, thread_pool.get(),
        (void *)ThreadPool::static_run);

    runtime_jit->call<void *, void *>("LLVMRuntime_set_assert_failed",
                                      llvm_runtime, (void *)assert_failed_host);
  }
  if (arch_is_cpu(config.arch)) {
    // Profiler functions can only be called on CPU kernels
    runtime_jit->call<void *, void *>("LLVMRuntime_set_profiler", llvm_runtime,
                                      profiler.get());
    runtime_jit->call<void *, void *>(
        "LLVMRuntime_set_profiler_start", llvm_runtime,
        (void *)&KernelProfilerBase::profiler_start);
    runtime_jit->call<void *, void *>(
        "LLVMRuntime_set_profiler_stop", llvm_runtime,
        (void *)&KernelProfilerBase::profiler_stop);
  }
}

void Program::initialize_llvm_runtime_snodes(const SNodeTree *tree,
                                             StructCompiler *scomp) {
  TaichiLLVMContext *tlctx = nullptr;
  if (is_cuda_no_unified_memory(config)) {
#if defined(TI_WITH_CUDA)
    tlctx = llvm_context_device.get();
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    tlctx = llvm_context_host.get();
  }
  auto *const runtime_jit = tlctx->runtime_jit_module;
  // By the time this creator is called, "this" is already destroyed.
  // Therefore it is necessary to capture members by values.
  const auto snodes = scomp->snodes;
  const int root_id = tree->root()->id;

  TI_TRACE("Allocating data structure of size {} bytes", scomp->root_size);
  runtime_jit->call<void *, std::size_t, int, int>(
      "runtime_initialize_snodes", llvm_runtime, scomp->root_size, root_id,
      (int)snodes.size(), tree->id());
  for (int i = 0; i < (int)snodes.size(); i++) {
    if (is_gc_able(snodes[i]->type)) {
      std::size_t node_size;
      auto element_size = snodes[i]->cell_size_bytes;
      if (snodes[i]->type == SNodeType::pointer) {
        // pointer. Allocators are for single elements
        node_size = element_size;
      } else {
        // dynamic. Allocators are for the chunks
        node_size = sizeof(void *) + element_size * snodes[i]->chunk_size;
      }
      TI_TRACE("Initializing allocator for snode {} (node size {})",
               snodes[i]->id, node_size);
      auto rt = llvm_runtime;
      runtime_jit->call<void *, int, std::size_t>(
          "runtime_NodeAllocator_initialize", rt, snodes[i]->id, node_size);
      TI_TRACE("Allocating ambient element for snode {} (node size {})",
               snodes[i]->id, node_size);
      runtime_jit->call<void *, int>("runtime_allocate_ambient", rt, i,
                                     node_size);
    }
  }
}

int Program::add_snode_tree(std::unique_ptr<SNode> root) {
  const int id = snode_trees_.size();
  auto tree = std::make_unique<SNodeTree>(id, std::move(root), config.packed);
  tree->root()->set_snode_tree_id(id);
  materialize_snode_tree(tree.get());
  snode_trees_.push_back(std::move(tree));
  return id;
}

SNode *Program::get_snode_root(int tree_id) {
  return snode_trees_[tree_id]->root();
}

std::unique_ptr<llvm::Module> Program::clone_struct_compiler_initial_context(
    TaichiLLVMContext *tlctx) {
  if (!snode_trees_.empty())
    return tlctx->clone_struct_module();
  return tlctx->clone_runtime_module();
}

void Program::materialize_snode_tree(SNodeTree *tree) {
  auto *const root = tree->root();
  // always use host_arch() for host accessors
  auto host_module =
      clone_struct_compiler_initial_context(llvm_context_host.get());
  std::unique_ptr<StructCompiler> scomp = std::make_unique<StructCompilerLLVM>(
      host_arch(), this, std::move(host_module));
  scomp->run(*root);
  materialize_snode_expr_attributes();

  for (auto snode : scomp->snodes) {
    snodes[snode->id] = snode;
  }

  if (arch_is_cpu(config.arch)) {
    initialize_llvm_runtime_snodes(tree, scomp.get());
  } else if (config.arch == Arch::cuda) {
    auto device_module =
        clone_struct_compiler_initial_context(llvm_context_device.get());

    std::unique_ptr<StructCompiler> scomp_gpu =
        std::make_unique<StructCompilerLLVM>(Arch::cuda, this,
                                             std::move(device_module));
    scomp_gpu->run(*root);
    initialize_llvm_runtime_snodes(tree, scomp_gpu.get());
  } else if (config.arch == Arch::metal) {
    TI_ASSERT_INFO(config.use_llvm,
                   "Metal arch requires that LLVM being enabled");
    metal_compiled_structs_ = metal::compile_structs(*root);
    if (metal_kernel_mgr_ == nullptr) {
      TI_ASSERT(result_buffer == nullptr);
      result_buffer = allocate_result_buffer_default(this);

      metal::KernelManager::Params params;
      params.compiled_structs = metal_compiled_structs_.value();
      params.config = &config;
      params.mem_pool = memory_pool.get();
      params.host_result_buffer = result_buffer;
      params.profiler = profiler.get();
      params.root_id = root->id;
      metal_kernel_mgr_ =
          std::make_unique<metal::KernelManager>(std::move(params));
    }
  } else if (config.arch == Arch::opengl) {
    TI_ASSERT(result_buffer == nullptr);
    result_buffer = allocate_result_buffer_default(this);
    opengl::OpenglStructCompiler scomp;
    opengl_struct_compiled_ = scomp.run(*root);
    TI_TRACE("OpenGL root buffer size: {} B",
             opengl_struct_compiled_->root_size);
    opengl_kernel_launcher_ = std::make_unique<opengl::GLSLLauncher>(
        opengl_struct_compiled_->root_size);
    opengl_kernel_launcher_->result_buffer = result_buffer;
#ifdef TI_WITH_CC
  } else if (config.arch == Arch::cc) {
    TI_ASSERT(result_buffer == nullptr);
    result_buffer = allocate_result_buffer_default(this);
    cc_program->compile_layout(root);
#endif
  }
}

void Program::check_runtime_error() {
  synchronize();
  auto tlctx = llvm_context_host.get();
  if (llvm_context_device) {
    // In case there is a standalone device context (e.g. CUDA without unified
    // memory), use the device context instead.
    tlctx = llvm_context_device.get();
  }
  auto *runtime_jit_module = tlctx->runtime_jit_module;
  runtime_jit_module->call<void *>("runtime_retrieve_and_reset_error_code",
                                   llvm_runtime);
  auto error_code = fetch_result<int64>(taichi_result_buffer_error_id);

  if (error_code) {
    std::string error_message_template;

    // Here we fetch the error_message_template char by char.
    // This is not efficient, but fortunately we only need to do this when an
    // assertion fails. Note that we may not have unified memory here, so using
    // "fetch_result" that works across device/host memroy is necessary.
    for (int i = 0;; i++) {
      runtime_jit_module->call<void *>("runtime_retrieve_error_message",
                                       llvm_runtime, i);
      auto c = fetch_result<char>(taichi_result_buffer_error_id);
      error_message_template += c;
      if (c == '\0') {
        break;
      }
    }

    if (error_code == 1) {
      const auto error_message_formatted = format_error_message(
          error_message_template, [runtime_jit_module, this](int argument_id) {
            runtime_jit_module->call<void *>(
                "runtime_retrieve_error_message_argument", llvm_runtime,
                argument_id);
            return fetch_result<uint64>(taichi_result_buffer_error_id);
          });
      TI_ERROR("Assertion failure: {}", error_message_formatted);
    } else {
      TI_NOT_IMPLEMENTED
    }
  }
}

void Program::synchronize() {
  if (!sync) {
    if (config.async_mode) {
      async_engine->synchronize();
    }
    if (profiler)
      profiler->sync();
    device_synchronize();
    sync = true;
  }
}

void Program::device_synchronize() {
  if (config.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().stream_synchronize(nullptr);
#else
    TI_ERROR("No CUDA support");
#endif
  } else if (config.arch == Arch::metal) {
    metal_kernel_mgr_->synchronize();
  }
}

void Program::async_flush() {
  if (!config.async_mode) {
    TI_WARN("No point calling async_flush() when async mode is disabled.");
    return;
  }
  async_engine->flush();
}

int Program::get_snode_tree_size() {
  return snode_trees_.size();
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
      for (int i = 0; i < taichi_max_num_indices; i++) {
        if (snode->extractors[i].active) {
          int nb = snode->extractors[i].num_bits;
          indices += fmt::format(
              R"($\mathbf{{{}}}^{{\mathbf{{{}b}}:{}}}_{{\mathbf{{{}b}}:{}}}$)",
              std::string(1, 'I' + i), 0, latex_short_digit(1 << 0), nb,
              latex_short_digit(1 << nb));
        }
      }
      if (!indices.empty())
        emit("\\\\" + indices);
      if (snode->type == SNodeType::place) {
        emit("\\\\" + data_type_name(snode->dt));
      }
      emit("} ");

      for (int i = 0; i < (int)snode->ch.size(); i++) {
        visit(snode->ch[i].get());
      }
      emit("]");
    };

    for (auto &a : snode_trees_) {
      visit(a->root());
    }

    auto tail = R"(
\end{tikzpicture}
\end{document}
)";
    emit(tail);
  }
  trash(system(fmt::format("pdflatex {}", fn).c_str()));
}

void Program::maybe_initialize_cuda_llvm_context() {
  if ((config.arch == Arch::cuda) && (llvm_context_device == nullptr)) {
    llvm_context_device = std::make_unique<TaichiLLVMContext>(Arch::cuda);
    llvm_context_device->init_runtime_jit_module();
  }
}

Arch Program::get_snode_accessor_arch() {
  if (config.arch == Arch::opengl) {
    return Arch::opengl;
  } else if (is_cuda_no_unified_memory(config)) {
    return Arch::cuda;
  } else if (config.arch == Arch::metal) {
    return Arch::metal;
  } else if (config.arch == Arch::cc) {
    return Arch::cc;
  } else {
    return get_host_arch();
  }
}

Kernel &Program::get_snode_reader(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_reader_{}", snode->id);
  auto &ker = kernel([snode, this] {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      indices.push_back(Expr::make<ArgLoadExpression>(i, PrimitiveType::i32));
    }
    auto ret = Stmt::make<FrontendReturnStmt>(
        load_if_ptr(Expr(snode_to_glb_var_exprs_.at(snode))[indices]));
    current_ast_builder().insert(std::move(ret));
  });
  ker.set_arch(get_snode_accessor_arch());
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_arg(PrimitiveType::i32, false);
  ker.insert_ret(snode->dt);
  return ker;
}

Kernel &Program::get_snode_writer(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_writer_{}", snode->id);
  auto &ker = kernel([snode, this] {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      indices.push_back(Expr::make<ArgLoadExpression>(i, PrimitiveType::i32));
    }
    Expr(snode_to_glb_var_exprs_.at(snode))[indices] =
        Expr::make<ArgLoadExpression>(snode->num_active_indices,
                                      snode->dt->get_compute_type());
  });
  ker.set_arch(get_snode_accessor_arch());
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_arg(PrimitiveType::i32, false);
  ker.insert_arg(snode->dt, false);
  return ker;
}

uint64 Program::fetch_result_uint64(int i) {
  // TODO: We are likely doing more synchronization than necessary. Simplify the
  // sync logic when we fetch the result.
  device_synchronize();
  uint64 ret;
  auto arch = config.arch;
  if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    if (config.use_unified_memory) {
      // More efficient than a cudaMemcpy call in practice
      ret = result_buffer[i];
    } else {
      CUDADriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i,
                                                       sizeof(uint64));
    }
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    ret = result_buffer[i];
  }
  return ret;
}

void Program::finalize() {
  synchronize();
  if (async_engine)
    async_engine = nullptr;  // Finalize the async engine threads before
                             // anything else gets destoried.
  TI_TRACE("Program finalizing...");
  if (config.print_benchmark_stat) {
    const char *current_test = std::getenv("PYTEST_CURRENT_TEST");
    const char *output_dir = std::getenv("TI_BENCHMARK_OUTPUT_DIR");
    if (current_test != nullptr) {
      if (output_dir == nullptr)
        output_dir = ".";
      std::string file_name = current_test;
      auto slash_pos = file_name.find_last_of('/');
      if (slash_pos != std::string::npos)
        file_name = file_name.substr(slash_pos + 1);
      auto py_pos = file_name.find(".py::");
      TI_ASSERT(py_pos != std::string::npos);
      file_name =
          file_name.substr(0, py_pos) + "__" + file_name.substr(py_pos + 5);
      auto first_space_pos = file_name.find_first_of(' ');
      TI_ASSERT(first_space_pos != std::string::npos);
      file_name = file_name.substr(0, first_space_pos);
      if (auto lt_pos = file_name.find('<'); lt_pos != std::string::npos) {
        file_name[lt_pos] = '_';
      }
      if (auto gt_pos = file_name.find('>'); gt_pos != std::string::npos) {
        file_name[gt_pos] = '_';
      }
      file_name += ".dat";
      file_name = std::string(output_dir) + "/" + file_name;
      TI_INFO("Saving benchmark result to {}", file_name);
      std::ofstream ofs(file_name);
      TI_ASSERT(ofs);
      std::string stat_string;
      stat.print(&stat_string);
      ofs << stat_string;
    }
  }
  if (runtime_mem_info)
    runtime_mem_info->set_profiler(nullptr);
  synchronize();
  current_program = nullptr;
  memory_pool->terminate();
#if defined(TI_WITH_CUDA)
  if (preallocated_device_buffer != nullptr)
    CUDADriver::get_instance().mem_free(preallocated_device_buffer);
#endif
  finalized = true;
  num_instances -= 1;
  TI_TRACE("Program ({}) finalized.", fmt::ptr(this));
}

int Program::default_block_dim(const CompileConfig &config) {
  if (arch_is_cpu(config.arch)) {
    return config.default_cpu_block_dim;
  } else {
    return config.default_gpu_block_dim;
  }
}

void Program::print_list_manager_info(void *list_manager) {
  auto list_manager_len =
      runtime_query<int32>("ListManager_get_num_elements", list_manager);

  auto element_size =
      runtime_query<int32>("ListManager_get_element_size", list_manager);

  auto elements_per_chunk = runtime_query<int32>(
      "ListManager_get_max_num_elements_per_chunk", list_manager);

  auto num_active_chunks =
      runtime_query<int32>("ListManager_get_num_active_chunks", list_manager);

  auto size_MB = 1e-6f * num_active_chunks * elements_per_chunk * element_size;

  fmt::print(
      " length={:n}     {:n} chunks x [{:n} x {:n} B]  total={:.4f} MB\n",
      list_manager_len, num_active_chunks, elements_per_chunk, element_size,
      size_MB);
}

void Program::print_memory_profiler_info() {
  TI_ASSERT(arch_uses_llvm(config.arch));

  fmt::print("\n[Memory Profiler]\n");

  std::locale::global(std::locale("en_US.UTF-8"));
  // So that thousand separators are added to "{:n}" slots in fmtlib.
  // E.g., 10000 is printed as "10,000".
  // TODO: is there a way to set locale only locally in this function?

  std::function<void(SNode *, int)> visit = [&](SNode *snode, int depth) {
    auto element_list = runtime_query<void *>("LLVMRuntime_get_element_lists",
                                              llvm_runtime, snode->id);

    if (snode->type != SNodeType::place) {
      fmt::print("SNode {:10}\n", snode->get_node_type_name_hinted());

      if (element_list) {
        fmt::print("  active element list:");
        print_list_manager_info(element_list);

        auto node_allocator = runtime_query<void *>(
            "LLVMRuntime_get_node_allocators", llvm_runtime, snode->id);

        if (node_allocator) {
          auto free_list = runtime_query<void *>("NodeManager_get_free_list",
                                                 node_allocator);
          auto recycled_list = runtime_query<void *>(
              "NodeManager_get_recycled_list", node_allocator);

          auto free_list_len =
              runtime_query<int32>("ListManager_get_num_elements", free_list);

          auto recycled_list_len = runtime_query<int32>(
              "ListManager_get_num_elements", recycled_list);

          auto free_list_used = runtime_query<int32>(
              "NodeManager_get_free_list_used", node_allocator);

          auto data_list = runtime_query<void *>("NodeManager_get_data_list",
                                                 node_allocator);
          fmt::print("  data list:          ");
          print_list_manager_info(data_list);

          fmt::print(
              "  Allocated elements={:n}; free list length={:n}; recycled list "
              "length={:n}\n",
              free_list_used, free_list_len, recycled_list_len);
        }
      }
    }
    for (const auto &ch : snode->ch) {
      visit(ch.get(), depth + 1);
    }
  };

  for (auto &a : snode_trees_) {
    visit(a->root(), /*depth=*/0);
  }

  auto total_requested_memory = runtime_query<std::size_t>(
      "LLVMRuntime_get_total_requested_memory", llvm_runtime);

  fmt::print(
      "Total requested dynamic memory (excluding alignment padding): {:n} B\n",
      total_requested_memory);
}

std::size_t Program::get_snode_num_dynamically_allocated(SNode *snode) {
  if (config.arch == Arch::metal) {
    return metal_kernel_mgr_->get_snode_num_dynamically_allocated(snode);
  }
  auto node_allocator = runtime_query<void *>("LLVMRuntime_get_node_allocators",
                                              llvm_runtime, snode->id);
  auto data_list =
      runtime_query<void *>("NodeManager_get_data_list", node_allocator);

  return (std::size_t)runtime_query<int32>("ListManager_get_num_elements",
                                           data_list);
}

Program::~Program() {
  if (!finalized)
    finalize();
}

void Program::materialize_snode_expr_attributes() {
  for (auto &[snode, glb_var] : snode_to_glb_var_exprs_) {
    glb_var->set_attribute("dim", std::to_string(snode->num_active_indices));
  }
}

std::unique_ptr<AotModuleBuilder> Program::make_aot_module_builder(Arch arch) {
  if (arch == Arch::metal) {
    return std::make_unique<metal::AotModuleBuilderImpl>(
        &(metal_compiled_structs_.value()),
        metal_kernel_mgr_->get_buffer_meta_data());
  } else if (arch == Arch::wasm) {
    return std::make_unique<wasm::AotModuleBuilderImpl>();
  }
  return nullptr;
}

}  // namespace lang
}  // namespace taichi
