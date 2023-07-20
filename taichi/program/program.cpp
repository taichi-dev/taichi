// Program, context for Taichi program execution

#include "program.h"

#include "taichi/ir/statements.h"
#include "taichi/program/extension.h"
#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/struct/struct.h"
#include "taichi/runtime/program_impls/opengl/opengl_program.h"
#include "taichi/runtime/program_impls/metal/metal_program.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/system/timeline.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/math/arithmetic.h"
#include "taichi/rhi/common/host_memory_pool.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/codegen/llvm/struct_llvm.h"
#endif

#ifdef TI_WITH_VULKAN
#include "taichi/runtime/program_impls/vulkan/vulkan_program.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif
#ifdef TI_WITH_OPENGL
#include "taichi/runtime/program_impls/opengl/opengl_program.h"
#include "taichi/rhi/opengl/opengl_api.h"
#endif
#ifdef TI_WITH_DX11
#include "taichi/runtime/program_impls/dx/dx_program.h"
#include "taichi/rhi/dx/dx_api.h"
#endif
#ifdef TI_WITH_DX12
#include "taichi/runtime/program_impls/dx12/dx12_program.h"
#include "taichi/rhi/dx12/dx12_api.h"
#endif
#ifdef TI_WITH_METAL
#include "taichi/runtime/program_impls/metal/metal_program.h"
#include "taichi/rhi/metal/metal_api.h"
#endif  // TI_WITH_METAL

#if defined(_M_X64) || defined(__x86_64)
// For _MM_SET_FLUSH_ZERO_MODE
#include <xmmintrin.h>
#endif  // defined(_M_X64) || defined(__x86_64)

namespace taichi::lang {
std::atomic<int> Program::num_instances_;

Program::Program(Arch desired_arch) : snode_rw_accessors_bank_(this) {
  TI_TRACE("Program initializing...");

  // For performance considerations and correctness of QuantFloatType
  // operations, we force floating-point operations to flush to zero on all
  // backends (including CPUs).
#if defined(_M_X64) || defined(__x86_64)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif  // defined(_M_X64) || defined(__x86_64)
#if defined(__arm64__) || defined(__aarch64__)
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
#endif  // defined(__arm64__) || defined(__aarch64__)
  auto &config = compile_config_;
  config = default_compile_config;
  config.arch = desired_arch;
  config.fit();

  profiler = make_profiler(config.arch, config.kernel_profiler);
  if (arch_uses_llvm(config.arch)) {
#ifdef TI_WITH_LLVM
    if (config.arch != Arch::dx12) {
      program_impl_ = std::make_unique<LlvmProgramImpl>(config, profiler.get());
    } else {
      // NOTE: use Dx12ProgramImpl to avoid using LlvmRuntimeExecutor for dx12.
#ifdef TI_WITH_DX12
      TI_ASSERT(directx12::is_dx12_api_available());
      program_impl_ = std::make_unique<Dx12ProgramImpl>(config);
#else
      TI_ERROR("This taichi is not compiled with DX12");
#endif
    }
#else
    TI_ERROR("This taichi is not compiled with LLVM");
#endif
  } else if (config.arch == Arch::metal) {
#ifdef TI_WITH_METAL
    TI_ASSERT(metal::is_metal_api_available());
    program_impl_ = std::make_unique<MetalProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with Metal")
#endif
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
#ifdef TI_WITH_OPENGL
    TI_ASSERT(opengl::initialize_opengl(false));
    program_impl_ = std::make_unique<OpenglProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with OpenGL");
#endif
  } else if (config.arch == Arch::gles) {
#ifdef TI_WITH_OPENGL
    TI_ASSERT(opengl::initialize_opengl(true));
    program_impl_ = std::make_unique<OpenglProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with OpenGL");
#endif
  } else {
    TI_NOT_IMPLEMENTED
  }

  // program_impl_ should be set in the if-else branch above
  TI_ASSERT(program_impl_);

  Device *compute_device = nullptr;
  compute_device = program_impl_->get_compute_device();
  // Must have handled all the arch fallback logic by this point.
  TI_ASSERT_INFO(num_instances_ == 0, "Only one instance at a time");
  total_compilation_time_ = 0;
  num_instances_ += 1;
  SNode::counter = 0;

  result_buffer = nullptr;
  finalized_ = false;

  if (!is_extension_supported(config.arch, Extension::assertion)) {
    if (config.check_out_of_bound) {
      TI_WARN("Out-of-bound access checking is not supported on arch={}",
              arch_name(config.arch));
      config.check_out_of_bound = false;
    }
  }

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

const CompiledKernelData &Program::compile_kernel(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  auto start_t = Time::get_time();
  TI_AUTO_PROF;
  auto &mgr = program_impl_->get_kernel_compilation_manager();
  const auto &ckd = mgr.load_or_compile(compile_config, caps, kernel_def);
  total_compilation_time_ += Time::get_time() - start_t;
  return ckd;
}

void Program::launch_kernel(const CompiledKernelData &compiled_kernel_data,
                            LaunchContextBuilder &ctx) {
  program_impl_->get_kernel_launcher().launch_kernel(compiled_kernel_data, ctx);
  if (compile_config().debug && arch_uses_llvm(compiled_kernel_data.arch())) {
    program_impl_->check_runtime_error(result_buffer);
  }
}

void Program::materialize_runtime() {
  program_impl_->materialize_runtime(profiler.get(), &result_buffer);
}

static void remove_rw_accessor_cache(
    SNode *parent_snode,
    SNodeRwAccessorsBank *snode_rw_accessors_bank) {
  for (int i = 0; i < (int)parent_snode->ch.size(); i++) {
    auto child_snode = parent_snode->ch[i].get();
    if (child_snode->type == SNodeType::place) {
      snode_rw_accessors_bank->remove_cached_kernels(child_snode);
    }
    remove_rw_accessor_cache(child_snode, snode_rw_accessors_bank);
  }
}

void Program::destroy_snode_tree(SNodeTree *snode_tree) {
  TI_ASSERT(arch_uses_llvm(compile_config().arch) ||
            compile_config().arch == Arch::vulkan ||
            compile_config().arch == Arch::dx11 ||
            compile_config().arch == Arch::dx12);

  // When accessing a ti.field at Python scope, SNodeRwAccessorsBank creates
  // a Taichi Kernel to read/write the field in a JIT manner, which caches the
  // compiled JIT Kernel so as to avoid recompilation when accessing the same
  // field.

  // This cache uses the place-SNode's address (SNode*) as the key,
  // which becomes unsafe once the SNodeTree gets destroyed and that
  // place-SNode's address gets reused by another SNode. We have to remove all
  // cached kernels upon SNodeTree destruction.
  SNode *root = snode_tree->root();

  // Traverse SNodeTree to remove all cached RWAccessor kernels
  remove_rw_accessor_cache(root, &snode_rw_accessors_bank_);

  program_impl_->destroy_snode_tree(snode_tree);
  free_snode_tree_ids_.push(snode_tree->id());
}

SNodeTree *Program::add_snode_tree(std::unique_ptr<SNode> root,
                                   bool compile_only) {
  const int id = allocate_snode_tree_id();
  auto tree = std::make_unique<SNodeTree>(id, std::move(root));
  tree->root()->set_snode_tree_id(id);
  if (compile_only) {
    program_impl_->compile_snode_tree_types(tree.get());
  } else {
    program_impl_->materialize_snode_tree(tree.get(), result_buffer);
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

void Program::synchronize() {
  program_impl_->synchronize();
}

StreamSemaphore Program::flush() {
  return program_impl_->flush();
}

int Program::get_snode_tree_size() {
  return snode_trees_.size();
}

Kernel &Program::get_snode_reader(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_reader_{}", snode->id);
  auto &ker = kernel([snode, this](Kernel *kernel) {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      auto argload_expr = Expr::make<ArgLoadExpression>(std::vector<int>{i},
                                                        PrimitiveType::i32);
      argload_expr->type_check(&this->compile_config());
      indices.push_back(std::move(argload_expr));
    }
    ASTBuilder &builder = kernel->context->builder();
    auto ret = Stmt::make<FrontendReturnStmt>(ExprGroup(
        builder.expr_subscript(Expr(snode_to_fields_.at(snode)), indices)));
    builder.insert(std::move(ret));
  });
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_scalar_param(PrimitiveType::i32);
  ker.insert_ret(snode->dt);
  ker.finalize_params();
  ker.finalize_rets();
  return ker;
}

Kernel &Program::get_snode_writer(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_writer_{}", snode->id);
  auto &ker = kernel([snode, this](Kernel *kernel) {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      auto argload_expr = Expr::make<ArgLoadExpression>(std::vector<int>{i},
                                                        PrimitiveType::i32);
      argload_expr->type_check(&this->compile_config());
      indices.push_back(std::move(argload_expr));
    }
    ASTBuilder &builder = kernel->context->builder();
    auto expr =
        builder.expr_subscript(Expr(snode_to_fields_.at(snode)), indices);
    auto argload_expr = Expr::make<ArgLoadExpression>(
        std::vector<int>{snode->num_active_indices},
        snode->dt->get_compute_type());
    argload_expr->type_check(&this->compile_config());
    builder.insert_assignment(expr, argload_expr, expr->dbg_info);
  });
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_scalar_param(PrimitiveType::i32);
  ker.insert_scalar_param(snode->dt);
  ker.finalize_params();
  ker.finalize_rets();
  return ker;
}

uint64 Program::fetch_result_uint64(int i) {
  return program_impl_->fetch_result_uint64(i, result_buffer);
}

void Program::finalize() {
  if (finalized_) {
    return;
  }

  synchronize();
  TI_TRACE("Program finalizing...");

  synchronize();
  if (arch_uses_llvm(compile_config().arch)) {
    program_impl_->finalize();
  }

  Stmt::reset_counter();

  finalized_ = true;
  num_instances_ -= 1;
  program_impl_->dump_cache_data_to_disk();
  compile_config_ = default_compile_config;
  TI_TRACE("Program ({}) finalized_.", fmt::ptr(this));

  // Reset memory pool
  HostMemoryPool::get_instance().reset();
}

int Program::default_block_dim(const CompileConfig &config) {
  if (arch_is_cpu(config.arch)) {
    return config.default_cpu_block_dim;
  } else {
    return config.default_gpu_block_dim;
  }
}

void Program::print_memory_profiler_info() {
  program_impl_->print_memory_profiler_info(snode_trees_, result_buffer);
}

std::size_t Program::get_snode_num_dynamically_allocated(SNode *snode) {
  return program_impl_->get_snode_num_dynamically_allocated(snode,
                                                            result_buffer);
}

Ndarray *Program::create_ndarray(const DataType type,
                                 const std::vector<int> &shape,
                                 ExternalArrayLayout layout,
                                 bool zero_fill,
                                 const DebugInfo &dbg_info) {
  auto arr = std::make_unique<Ndarray>(this, type, shape, layout, dbg_info);
  if (zero_fill) {
    Arch arch = compile_config().arch;
    if (arch_is_cpu(arch) || arch == Arch::cuda || arch == Arch::amdgpu) {
      fill_ndarray_fast_u32(arr.get(), /*data=*/0);
    } else if (arch != Arch::dx12) {
      // Device api support for dx12 backend are not complete yet
      Stream *stream =
          program_impl_->get_compute_device()->get_compute_stream();
      auto [cmdlist, res] = stream->new_command_list_unique();
      TI_ASSERT(res == RhiResult::success);
      cmdlist->buffer_fill(arr->ndarray_alloc_.get_ptr(0),
                           arr->get_element_size() * arr->get_nelement(),
                           /*data=*/0);
      stream->submit_synced(cmdlist.get());
    }
  }
  auto arr_ptr = arr.get();
  ndarrays_.insert({arr_ptr, std::move(arr)});
  return arr_ptr;
}

ArgPack *Program::create_argpack(const DataType dt) {
  auto pack = std::make_unique<ArgPack>(this, dt);
  auto pack_ptr = pack.get();
  argpacks_.insert({pack_ptr, std::move(pack)});
  return pack_ptr;
}

void Program::delete_ndarray(Ndarray *ndarray) {
  // [Note] Ndarray memory deallocation
  // Ndarray's memory allocation is managed by Taichi and Python can control
  // this via Taichi indirectly. For example, when an ndarray is GC-ed in
  // Python, it signals Taichi to free its memory allocation. But Taichi will
  // make sure **no pending kernels to be executed needs the ndarray** before it
  // actually frees the memory. When `ti.reset()` is called, all ndarrays
  // allocated in this program should be gone and no longer valid in Python.
  // This isn't the best implementation, ndarrays should be managed by taichi
  // runtime instead of this giant program and it should be freed when:
  // - Python GC signals taichi that it's no longer useful
  // - All kernels using it are executed.
  if (ndarrays_.count(ndarray) &&
      !program_impl_->used_in_kernel(ndarray->ndarray_alloc_.alloc_id)) {
    ndarrays_.erase(ndarray);
  }
}

void Program::delete_argpack(ArgPack *argpack) {
  // [Note] Argpack memory deallocation
  // Argpack's memory allocation is managed by Taichi and Python can control
  // this via Taichi indirectly. For example, when an argpack is GC-ed in
  // Python, it signals Taichi to free its memory allocation. But Taichi will
  // make sure **no pending kernels to be executed needs the argpack** before it
  // actually frees the memory. When `ti.reset()` is called, all argpack
  // allocated in this program should be gone and no longer valid in Python.
  // This isn't the best implementation, argpacks should be managed by taichi
  // runtime instead of this giant program and it should be freed when:
  // - Python GC signals taichi that it's no longer useful
  // - All kernels using it are executed.
  if (argpacks_.count(argpack) &&
      !program_impl_->used_in_kernel(argpack->argpack_alloc_.alloc_id)) {
    argpacks_.erase(argpack);
  }
}

Texture *Program::create_texture(BufferFormat buffer_format,
                                 const std::vector<int> &shape) {
  if (shape.size() == 1) {
    textures_.push_back(
        std::make_unique<Texture>(this, buffer_format, shape[0], 1, 1));
  } else if (shape.size() == 2) {
    textures_.push_back(
        std::make_unique<Texture>(this, buffer_format, shape[0], shape[1], 1));
  } else if (shape.size() == 3) {
    textures_.push_back(std::make_unique<Texture>(this, buffer_format, shape[0],
                                                  shape[1], shape[2]));
  } else {
    TI_ERROR("Texture shape invalid");
  }
  return textures_.back().get();
}

intptr_t Program::get_ndarray_data_ptr_as_int(const Ndarray *ndarray) {
  uint64_t *data_ptr{nullptr};
  if (arch_is_cpu(compile_config().arch) ||
      compile_config().arch == Arch::cuda ||
      compile_config().arch == Arch::amdgpu) {
    // For the LLVM backends, device allocation is a physical pointer.
    data_ptr =
        program_impl_->get_device_alloc_info_ptr(ndarray->ndarray_alloc_);
  }

  return reinterpret_cast<intptr_t>(data_ptr);
}

void Program::fill_ndarray_fast_u32(Ndarray *ndarray, uint32_t val) {
  // This is a temporary solution to bypass device api.
  // Should be moved to CommandList once available in CUDA.
  program_impl_->fill_ndarray(
      ndarray->ndarray_alloc_,
      ndarray->get_nelement() * ndarray->get_element_size() / sizeof(uint32_t),
      val);
}

std::pair<const ArgPackType *, size_t>
Program::get_argpack_type_with_data_layout(const ArgPackType *old_ty,
                                           const std::string &layout) {
  // Convert to StructType
  auto *struct_type_old =
      TypeFactory::get_instance()
          .get_struct_type(old_ty->elements(), old_ty->get_layout())
          ->as<StructType>();
  // Call get_struct_type_with_data_layout
  auto [struct_type, size] = program_impl_->get_struct_type_with_data_layout(
      const_cast<StructType *>(struct_type_old), layout);
  // Convert back to ArgPackType
  auto *new_ty =
      TypeFactory::get_instance()
          .get_argpack_type(struct_type->elements(), struct_type->get_layout())
          ->as<ArgPackType>();
  return {new_ty, size};
}

std::pair<const StructType *, size_t> Program::get_struct_type_with_data_layout(
    const StructType *old_ty,
    const std::string &layout) {
  return program_impl_->get_struct_type_with_data_layout(old_ty, layout);
}

Program::~Program() {
  finalize();
}

DeviceCapabilityConfig translate_devcaps(const std::vector<std::string> &caps) {
  // Each device capability assignment is named like this:
  // - `spirv_version=1.3`
  // - `spirv_has_int8`
  DeviceCapabilityConfig cfg{};
  for (const std::string &cap : caps) {
    std::string_view key;
    uint32_t value;
    size_t ieq = cap.find('=');
    if (ieq == std::string::npos) {
      key = cap;
      value = 1;
    } else {
      key = std::string_view(cap.c_str(), ieq);
      value = (uint32_t)std::atol(cap.c_str() + ieq + 1);
    }
    DeviceCapability devcap = str2devcap(key);
    cfg.set(devcap, value);
  }

  // Assign default caps (that always present).
  if (!cfg.contains(DeviceCapability::spirv_version)) {
    cfg.set(DeviceCapability::spirv_version, 0x10300);
  }
  return cfg;
}

std::unique_ptr<AotModuleBuilder> Program::make_aot_module_builder(
    Arch arch,
    const std::vector<std::string> &caps) {
  DeviceCapabilityConfig cfg = translate_devcaps(caps);
  // FIXME: This couples the runtime backend with the target AOT backend. E.g.
  // If we want to build a Metal AOT module, we have to be on the macOS
  // platform. Consider decoupling this part
  if (arch_uses_llvm(compile_config().arch) ||
      compile_config().arch == Arch::metal ||
      compile_config().arch == Arch::vulkan ||
      compile_config().arch == Arch::opengl ||
      compile_config().arch == Arch::gles ||
      compile_config().arch == Arch::dx12) {
    return program_impl_->make_aot_module_builder(cfg);
  }
  return nullptr;
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

void Program::enqueue_compute_op_lambda(
    std::function<void(Device *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs) {
  program_impl_->enqueue_compute_op_lambda(op, image_refs);
}

}  // namespace taichi::lang
