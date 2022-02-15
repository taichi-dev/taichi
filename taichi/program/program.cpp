// Program, context for Taichi program execution

#include "program.h"

#include "taichi/ir/statements.h"
#include "taichi/program/extension.h"
#include "taichi/backends/cpu/codegen_cpu.h"
#include "taichi/struct/struct.h"
#include "taichi/struct/struct_llvm.h"
#include "taichi/backends/metal/api.h"
#include "taichi/backends/wasm/aot_module_builder_impl.h"
#include "taichi/backends/opengl/opengl_program.h"
#include "taichi/backends/metal/metal_program.h"
#include "taichi/backends/cc/cc_program.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/system/unified_allocator.h"
#include "taichi/system/timeline.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/program/async_engine.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/util/statistics.h"
#include "taichi/math/arithmetic.h"
#ifdef TI_WITH_LLVM
#include "taichi/llvm/llvm_program.h"
#endif

#if defined(TI_WITH_CC)
#include "taichi/backends/cc/cc_program.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/backends/vulkan/vulkan_program.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#endif
#ifdef TI_WITH_DX11
#include "taichi/backends/dx/dx_program.h"
#include "taichi/backends/dx/dx_api.h"
#endif

#if defined(TI_ARCH_x64)
// For _MM_SET_FLUSH_ZERO_MODE
#include <xmmintrin.h>
#endif

namespace taichi {
namespace lang {
std::atomic<int> Program::num_instances_;

Program::Program(Arch desired_arch)
    : snode_rw_accessors_bank_(this), ndarray_rw_accessors_bank_(this) {
  TI_TRACE("Program initializing...");

  // For performance considerations and correctness of CustomFloatType
  // operations, we force floating-point operations to flush to zero on all
  // backends (including CPUs).
#if defined(TI_ARCH_x64)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#elif !defined(TI_EMSCRIPTENED)
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

  profiler = make_profiler(config.arch, config.kernel_profiler);
  if (arch_uses_llvm(config.arch)) {
#ifdef TI_WITH_LLVM
    program_impl_ = std::make_unique<LlvmProgramImpl>(config, profiler.get());
#else
    TI_ERROR("This taichi is not compiled with LLVM");
#endif
  } else if (config.arch == Arch::metal) {
    TI_ASSERT(metal::is_metal_api_available());
    program_impl_ = std::make_unique<MetalProgramImpl>(config);
  } else if (config.arch == Arch::vulkan) {
#ifdef TI_WITH_VULKAN
    TI_ASSERT(vulkan::is_vulkan_api_available());
    program_impl_ = std::make_unique<VulkanProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with Vulkan")
#endif
  } else if (config.arch == Arch::dx11) {
#ifdef TI_WITH_DX11
    TI_ASSERT(directx11::is_dx_api_available());
    program_impl_ = std::make_unique<Dx11ProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with DX11");
#endif
  } else if (config.arch == Arch::opengl) {
    TI_ASSERT(opengl::initialize_opengl(config.use_gles));
    program_impl_ = std::make_unique<OpenglProgramImpl>(config);
  } else if (config.arch == Arch::cc) {
#ifdef TI_WITH_CC
    program_impl_ = std::make_unique<CCProgramImpl>(config);
#else
    TI_ERROR("No C backend detected.");
#endif
  } else {
    TI_NOT_IMPLEMENTED
  }

  // program_impl_ should be set in the if-else branch above
  TI_ASSERT(program_impl_);

  Device *compute_device = nullptr;
  compute_device = program_impl_->get_compute_device();
  // Must have handled all the arch fallback logic by this point.
  memory_pool_ = std::make_unique<MemoryPool>(config.arch, compute_device);
  TI_ASSERT_INFO(num_instances_ == 0, "Only one instance at a time");
  total_compilation_time_ = 0;
  num_instances_ += 1;
  SNode::counter = 0;
  if (arch_uses_llvm(config.arch)) {
#if TI_WITH_LLVM
    static_cast<LlvmProgramImpl *>(program_impl_.get())->initialize_host();
#else
    TI_NOT_IMPLEMENTED
#endif
  }

  result_buffer = nullptr;
  current_callable = nullptr;
  sync = true;
  finalized_ = false;

  if (config.async_mode) {
    TI_WARN("Running in async mode. This is experimental.");
    TI_ASSERT(is_extension_supported(config.arch, Extension::async_mode));
    async_engine = std::make_unique<AsyncEngine>(
        &config, [this](Kernel &kernel, OffloadedStmt *offloaded) {
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
  auto ret = program_impl_->compile(&kernel, offloaded);
  TI_ASSERT(ret);
  total_compilation_time_ += Time::get_time() - start_t;
  return ret;
}

void Program::materialize_runtime() {
  program_impl_->materialize_runtime(memory_pool_.get(), profiler.get(),
                                     &result_buffer);
}

void Program::destroy_snode_tree(SNodeTree *snode_tree) {
  TI_ASSERT(arch_uses_llvm(config.arch) || config.arch == Arch::vulkan);
  program_impl_->destroy_snode_tree(snode_tree);
  free_snode_tree_ids_.push(snode_tree->id());
}

SNodeTree *Program::add_snode_tree(std::unique_ptr<SNode> root,
                                   bool compile_only) {
  const int id = allocate_snode_tree_id();
  auto tree = std::make_unique<SNodeTree>(id, std::move(root));
  tree->root()->set_snode_tree_id(id);
  if (compile_only) {
    program_impl_->compile_snode_tree_types(tree.get(), snode_trees_);
  } else {
    program_impl_->materialize_snode_tree(tree.get(), snode_trees_,
                                          result_buffer);
  }
  if (id < snode_trees_.size()) {
    snode_trees_[id] = std::move(tree);
  } else {
    TI_ASSERT(id == snode_trees_.size());
    snode_trees_.push_back(std::move(tree));
  }
  return snode_trees_[id].get();
}

SNode *Program::get_snode_root(int tree_id) {
  return snode_trees_[tree_id]->root();
}

void Program::check_runtime_error() {
#ifdef TI_WITH_LLVM
  TI_ASSERT(arch_uses_llvm(config.arch));
  static_cast<LlvmProgramImpl *>(program_impl_.get())
      ->check_runtime_error(result_buffer);
#else
  TI_ERROR("Llvm disabled");
#endif
}

void Program::synchronize() {
  if (!sync) {
    if (config.async_mode) {
      async_engine->synchronize();
    }
    if (arch_uses_llvm(config.arch) || config.arch == Arch::metal ||
        config.arch == Arch::vulkan) {
      program_impl_->synchronize();
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

Arch Program::get_accessor_arch() {
  if (config.arch == Arch::opengl) {
    return Arch::opengl;
  } else if (config.arch == Arch::vulkan) {
    return Arch::vulkan;
  } else if (config.arch == Arch::cuda) {
    return Arch::cuda;
  } else if (config.arch == Arch::metal) {
    return Arch::metal;
  } else if (config.arch == Arch::cc) {
    return Arch::cc;
  } else if (config.arch == Arch::dx11) {
    return Arch::dx11;
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
        ExprGroup(Expr(snode_to_glb_var_exprs_.at(snode))[indices]));
    this->current_ast_builder()->insert(std::move(ret));
  });
  ker.set_arch(get_accessor_arch());
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
    auto expr = Expr(snode_to_glb_var_exprs_.at(snode))[indices];
    this->current_ast_builder()->insert_assignment(
        expr, Expr::make<ArgLoadExpression>(snode->num_active_indices,
                                            snode->dt->get_compute_type()));
  });
  ker.set_arch(get_accessor_arch());
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_arg(PrimitiveType::i32, false);
  ker.insert_arg(snode->dt, false);
  return ker;
}

Kernel &Program::get_ndarray_reader(Ndarray *ndarray) {
  auto kernel_name = fmt::format("ndarray_reader");
  NdarrayRwKeys keys{ndarray->num_active_indices, ndarray->dtype};
  auto &ker = kernel([keys, this] {
    ExprGroup indices;
    for (int i = 0; i < keys.num_active_indices; i++) {
      indices.push_back(Expr::make<ArgLoadExpression>(i, PrimitiveType::i32));
    }
    auto ret = Stmt::make<FrontendReturnStmt>(
        ExprGroup(Expr(Expr::make<ExternalTensorExpression>(
            keys.dtype, keys.num_active_indices, keys.num_active_indices,
            0))[indices]));
    this->current_ast_builder()->insert(std::move(ret));
  });
  ker.set_arch(get_accessor_arch());
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < keys.num_active_indices; i++)
    ker.insert_arg(PrimitiveType::i32, false);
  ker.insert_arg(keys.dtype, true);
  ker.insert_ret(keys.dtype);
  return ker;
}

Kernel &Program::get_ndarray_writer(Ndarray *ndarray) {
  auto kernel_name = fmt::format("ndarray_writer");
  NdarrayRwKeys keys{ndarray->num_active_indices, ndarray->dtype};
  auto &ker = kernel([keys, this] {
    ExprGroup indices;
    for (int i = 0; i < keys.num_active_indices; i++) {
      indices.push_back(Expr::make<ArgLoadExpression>(i, PrimitiveType::i32));
    }
    auto expr = Expr(Expr::make<ExternalTensorExpression>(
        keys.dtype, keys.num_active_indices, keys.num_active_indices + 1,
        0))[indices];
    this->current_ast_builder()->insert_assignment(
        expr, Expr::make<ArgLoadExpression>(keys.num_active_indices,
                                            keys.dtype->get_compute_type()));
  });
  ker.set_arch(get_accessor_arch());
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < keys.num_active_indices; i++)
    ker.insert_arg(PrimitiveType::i32, false);
  ker.insert_arg(keys.dtype, false);
  ker.insert_arg(keys.dtype, true);
  return ker;
}

uint64 Program::fetch_result_uint64(int i) {
  if (arch_uses_llvm(config.arch)) {
#ifdef TI_WITH_LLVM
    return static_cast<LlvmProgramImpl *>(program_impl_.get())
        ->fetch_result<uint64>(i, result_buffer);
#else
    TI_NOT_IMPLEMENTED
#endif
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
  memory_pool_->terminate();

  if (arch_uses_llvm(config.arch)) {
#if TI_WITH_LLVM
    static_cast<LlvmProgramImpl *>(program_impl_.get())->finalize();
#else
    TI_NOT_IMPLEMENTED
#endif
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
#ifdef TI_WITH_LLVM
  TI_ASSERT(arch_uses_llvm(config.arch));
  static_cast<LlvmProgramImpl *>(program_impl_.get())
      ->print_memory_profiler_info(snode_trees_, result_buffer);
#else
  TI_ERROR("Llvm disabled");
#endif
}

std::size_t Program::get_snode_num_dynamically_allocated(SNode *snode) {
  TI_ASSERT(arch_uses_llvm(config.arch) || config.arch == Arch::metal ||
            config.arch == Arch::vulkan || config.arch == Arch::opengl);
  return program_impl_->get_snode_num_dynamically_allocated(snode,
                                                            result_buffer);
}

Program::~Program() {
  if (!finalized_)
    finalize();
}

std::unique_ptr<AotModuleBuilder> Program::make_aot_module_builder(Arch arch) {
  // FIXME: This couples the runtime backend with the target AOT backend. E.g.
  // If we want to build a Metal AOT module, we have to be on the macOS
  // platform. Consider decoupling this part
  if (arch == Arch::wasm) {
    // Have to check WASM first, or it dispatches to the LlvmProgramImpl.
#ifdef TI_WITH_LLVM
    return std::make_unique<wasm::AotModuleBuilderImpl>();
#else
    TI_NOT_IMPLEMENTED
#endif
  }
  if (arch_uses_llvm(config.arch) || config.arch == Arch::metal ||
      config.arch == Arch::vulkan || config.arch == Arch::opengl) {
    return program_impl_->make_aot_module_builder();
  }
  return nullptr;
}

LlvmProgramImpl *Program::get_llvm_program_impl() {
#ifdef TI_WITH_LLVM
  return static_cast<LlvmProgramImpl *>(program_impl_.get());
#else
  TI_ERROR("Llvm disabled");
#endif
}

int Program::allocate_snode_tree_id() {
  if (free_snode_tree_ids_.empty()) {
    return snode_trees_.size();
  } else {
    int id = free_snode_tree_ids_.top();
    free_snode_tree_ids_.pop();
    return id;
  }
}

}  // namespace lang
}  // namespace taichi
