// Program, context for Taichi program execution

#include "program.h"

#include "taichi/ir/statements.h"
#include "taichi/program/extension.h"
#include "taichi/backends/opengl/opengl_api.h"
#include "taichi/backends/opengl/codegen_opengl.h"
#include "taichi/backends/cpu/codegen_cpu.h"
#include "taichi/struct/struct.h"
#include "taichi/struct/struct_llvm.h"
#include "taichi/backends/metal/api.h"
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
#include "taichi/math/arithmetic.h"

#if defined(TI_WITH_CC)
#include "taichi/backends/cc/struct_cc.h"
#include "taichi/backends/cc/cc_layout.h"
#include "taichi/backends/cc/codegen_cc.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/backends/vulkan/snode_struct_compiler.h"
#include "taichi/backends/vulkan/codegen_vulkan.h"
#endif

#if defined(TI_ARCH_x64)
// For _MM_SET_FLUSH_ZERO_MODE
#include <xmmintrin.h>
#endif

namespace taichi {
namespace lang {
namespace {
inline uint64 *allocate_result_buffer_default(Program *prog) {
  return (uint64 *)prog->memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
}
}  // namespace

Program *current_program = nullptr;
std::atomic<int> Program::num_instances_;

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
  config = default_compile_config;
  config.arch = desired_arch;
  // TODO: allow users to run in debug mode without out-of-bound checks
  if (config.debug)
    config.check_out_of_bound = true;

  profiler = make_profiler(config.arch);
  llvm_program_ = std::make_unique<LlvmProgramImpl>(config, profiler.get());

  if (config.arch == Arch::metal) {
    if (!metal::is_metal_api_available()) {
      TI_WARN("No Metal API detected.");
      config.arch = host_arch();
    } else {
      metal_program_ = std::make_unique<MetalProgramImpl>(config);
    }
  }

  if (config.arch == Arch::opengl) {
    if (!opengl::is_opengl_api_available()) {
      TI_WARN("No OpenGL API detected.");
      config.arch = host_arch();
    }
  }

  if (config.arch == Arch::cc) {
#ifdef TI_WITH_CC
    cc_program = std::make_unique<cccp::CCProgram>(this);
#else
    TI_WARN("No C backend detected.");
    config.arch = host_arch();
#endif
  }

  if (config.arch != desired_arch) {
    TI_WARN("Falling back to {}", arch_name(config.arch));
  }

  memory_pool = std::make_unique<MemoryPool>(this);
  TI_ASSERT_INFO(num_instances_ == 0, "Only one instance at a time");
  total_compilation_time_ = 0;
  num_instances_ += 1;
  SNode::counter = 0;
  TI_ASSERT(current_program == nullptr);
  current_program = this;

  llvm_program_->initialize_host();

  result_buffer = nullptr;
  current_callable = nullptr;
  sync = true;
  finalized_ = false;

  if (config.async_mode) {
    TI_WARN("Running in async mode. This is experimental.");
    TI_ASSERT(is_extension_supported(config.arch, Extension::async_mode));
    async_engine = std::make_unique<AsyncEngine>(
        this, [this](Kernel &kernel, OffloadedStmt *offloaded) {
          return this->compile(kernel, offloaded);
        });
  }

  if (!is_extension_supported(config.arch, Extension::assertion)) {
    if (config.check_out_of_bound) {
      TI_WARN("Out-of-bound access checking is not supported on arch={}",
              arch_name(config.arch));
      config.check_out_of_bound = false;
    }
  }

  stat.clear();

  Timelines::get_instance().set_enabled(config.timeline);

  TI_TRACE("Program ({}) arch={} initialized.", fmt::ptr(this),
           arch_name(config.arch));
}

TypeFactory &Program::get_type_factory() {
  TI_WARN(
      "Program::get_type_factory() will be deprecated, Please use "
      "TypeFactory::get_instance()");
  return TypeFactory::get_instance();
}

Function *Program::create_function(const FunctionKey &func_key) {
  TI_TRACE("Creating function {}...", func_key.get_full_name());
  functions_.emplace_back(std::make_unique<Function>(this, func_key));
  TI_ASSERT(function_map_.count(func_key) == 0);
  function_map_[func_key] = functions_.back().get();
  return functions_.back().get();
}

FunctionType Program::compile(Kernel &kernel, OffloadedStmt *offloaded) {
  auto start_t = Time::get_time();
  TI_AUTO_PROF;
  FunctionType ret = nullptr;
  if (arch_uses_llvm(config.arch)) {
    return llvm_program_->compile(&kernel, offloaded);
  } else if (kernel.arch == Arch::metal) {
    return metal_program_->compile(&kernel, offloaded);
  } else if (kernel.arch == Arch::opengl) {
    opengl::OpenglCodeGen codegen(kernel.name, &opengl_struct_compiled_.value(),
                                  opengl_kernel_launcher_.get());
    ret = codegen.compile(kernel);
#ifdef TI_WITH_CC
  } else if (kernel.arch == Arch::cc) {
    ret = cccp::compile_kernel(&kernel);
#endif
#ifdef TI_WITH_VULKAN
  } else if (kernel.arch == Arch::vulkan) {
    vulkan::lower(&kernel);
    ret = vulkan::compile_to_executable(
        &kernel, &vulkan_compiled_structs_.value(), vulkan_runtime_.get());
#endif  // TI_WITH_VULKAN
  } else {
    TI_NOT_IMPLEMENTED;
  }
  TI_ASSERT(ret);
  total_compilation_time_ += Time::get_time() - start_t;
  return ret;
}

void Program::materialize_runtime() {
  if (arch_uses_llvm(config.arch)) {
    llvm_program_->materialize_runtime(memory_pool.get(), profiler.get(),
                                       &result_buffer);
  }
}

void Program::destroy_snode_tree(SNodeTree *snode_tree) {
  TI_ASSERT(arch_uses_llvm(config.arch));
  llvm_program_->destroy_snode_tree(snode_tree);
}

SNodeTree *Program::add_snode_tree(std::unique_ptr<SNode> root) {
  const int id = snode_trees_.size();
  auto tree = std::make_unique<SNodeTree>(id, std::move(root));
  tree->root()->set_snode_tree_id(id);
  materialize_snode_tree(tree.get());
  snode_trees_.push_back(std::move(tree));
  return snode_trees_[id].get();
}

SNode *Program::get_snode_root(int tree_id) {
  return snode_trees_[tree_id]->root();
}

void Program::materialize_snode_tree(SNodeTree *tree) {
  auto *const root = tree->root();
  if (arch_is_cpu(config.arch) || config.arch == Arch::cuda) {
    llvm_program_->materialize_snode_tree(
        tree, snode_trees_, snodes, snode_to_glb_var_exprs_, result_buffer);
  } else if (config.arch == Arch::metal) {
    metal_program_->materialize_snode_tree(tree, &result_buffer,
                                           memory_pool.get(), profiler.get());
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
  } else if (config.arch == Arch::vulkan) {
#ifdef TI_WITH_VULKAN
    result_buffer = allocate_result_buffer_default(this);
    vulkan_compiled_structs_ = vulkan::compile_snode_structs(*root);
    vulkan::VkRuntime::Params params;
    params.snode_descriptors = &(vulkan_compiled_structs_->snode_descriptors);
    params.host_result_buffer = result_buffer;
    vulkan_runtime_ = std::make_unique<vulkan::VkRuntime>(std::move(params));
#endif
  }
}

void Program::check_runtime_error() {
  TI_ASSERT(arch_uses_llvm(config.arch));
  llvm_program_->check_runtime_error(result_buffer);
}

void Program::synchronize() {
  if (!sync) {
    if (config.async_mode) {
      async_engine->synchronize();
    }
    if (profiler) {
      profiler->sync();
    }
    if (arch_uses_llvm(config.arch)) {
      llvm_program_->synchronize();
    } else if (config.arch == Arch::metal) {
      metal_program_->synchronize();
    } else if (config.arch == Arch::vulkan) {
      vulkan_runtime_->synchronize();
    }
    sync = true;
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

Arch Program::get_snode_accessor_arch() {
  if (config.arch == Arch::opengl) {
    return Arch::opengl;
  } else if (config.arch == Arch::vulkan) {
    return Arch::vulkan;
  } else if (config.is_cuda_no_unified_memory()) {
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
  if (arch_uses_llvm(config.arch)) {
    return llvm_program_->fetch_result<uint64>(i, result_buffer);
  }
  return result_buffer[i];
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

  synchronize();
  current_program = nullptr;
  memory_pool->terminate();

  if (arch_uses_llvm(config.arch)) {
    llvm_program_->finalize();
  }

  finalized_ = true;
  num_instances_ -= 1;
  TI_TRACE("Program ({}) finalized_.", fmt::ptr(this));
}

int Program::default_block_dim(const CompileConfig &config) {
  if (arch_is_cpu(config.arch)) {
    return config.default_cpu_block_dim;
  } else {
    return config.default_gpu_block_dim;
  }
}

void Program::print_memory_profiler_info() {
  TI_ASSERT(arch_uses_llvm(config.arch));
  llvm_program_->print_memory_profiler_info(snode_trees_, result_buffer);
}

std::size_t Program::get_snode_num_dynamically_allocated(SNode *snode) {
  if (config.arch == Arch::metal) {
    return metal_program_->get_snode_num_dynamically_allocated(snode);
  }
  return llvm_program_->get_snode_num_dynamically_allocated(snode,
                                                            result_buffer);
}

Program::~Program() {
  if (!finalized_)
    finalize();
}

std::unique_ptr<AotModuleBuilder> Program::make_aot_module_builder(Arch arch) {
  if (arch == Arch::metal) {
    return metal_program_->make_aot_module_builder();
  } else if (arch == Arch::wasm) {
    return std::make_unique<wasm::AotModuleBuilderImpl>();
  }
  return nullptr;
}

}  // namespace lang
}  // namespace taichi
