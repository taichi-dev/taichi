#include "taichi/backends/cc/cc_program.h"

using namespace taichi::lang::cccp;

TLANG_NAMESPACE_BEGIN

CCProgramImpl::CCProgramImpl(CompileConfig &config) : ProgramImpl(config) {
  this->config = &config;
  runtime_ = std::make_unique<CCRuntime>(this,
#include "runtime/base.h"
                                         "\n",
#include "runtime/base.c"
                                         "\n");
  runtime_->compile();
  context_ = std::make_unique<CCContext>();
}

FunctionType CCProgramImpl::compile(Kernel *kernel, OffloadedStmt *) {
  CCKernelGen codegen(kernel, this);
  auto ker = codegen.compile();
  auto ker_ptr = ker.get();
  this->add_kernel(std::move(ker));
  return [ker_ptr](RuntimeContext &ctx) { return ker_ptr->launch(&ctx); };
}

void CCProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                        KernelProfilerBase *,
                                        uint64 **result_buffer_ptr) {
  TI_ASSERT(*result_buffer_ptr == nullptr);
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  result_buffer_ = *result_buffer_ptr;
}

void CCProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                           uint64 *result_buffer) {
  auto *const root = tree->root();
  CCLayoutGen gen(this, root);
  layout_ = gen.compile();
  size_t root_size = layout_->compile();
  size_t gtmp_size = taichi_global_tmp_buffer_size;
  size_t args_size = taichi_result_buffer_entries * sizeof(uint64);

  TI_INFO("[cc] C backend root buffer size: {} B", root_size);

  ActionRecorder::get_instance().record(
      "allocate_buffer", {
                             ActionArg("root_size", (int32)root_size),
                             ActionArg("gtmp_size", (int32)gtmp_size),
                         });

  root_buf_.resize(root_size, 0);
  gtmp_buf_.resize(gtmp_size, 0);
  args_buf_.resize(args_size, 0);

  context_->root = root_buf_.data();
  context_->gtmp = gtmp_buf_.data();
  context_->args = (uint64 *)args_buf_.data();
  context_->earg = nullptr;
}

void CCProgramImpl::add_kernel(std::unique_ptr<CCKernel> kernel) {
  kernels_.push_back(std::move(kernel));
  need_relink_ = true;
}

void CCKernel::compile() {
  if (!kernel_->is_evaluator)
    ActionRecorder::get_instance().record(
        "compile_kernel", {
                              ActionArg("kernel_name", name_),
                              ActionArg("kernel_source", source_),
                          });

  obj_path_ = fmt::format("{}/{}.o", runtime_tmp_dir, name_);
  src_path_ = fmt::format("{}/{}.c", runtime_tmp_dir, name_);

  std::ofstream(src_path_) << cc_program_impl_->get_runtime()->header << "\n"
                           << cc_program_impl_->get_layout()->source << "\n"
                           << source_;
  TI_DEBUG("[cc] compiling [{}] -> [{}]:\n{}\n", name_, obj_path_, source_);
  execute(cc_program_impl_->config->cc_compile_cmd, obj_path_, src_path_);
}

void CCRuntime::compile() {
  ActionRecorder::get_instance().record("compile_runtime",
                                        {
                                            ActionArg("runtime_header", header),
                                            ActionArg("runtime_source", source),
                                        });

  obj_path_ = fmt::format("{}/_rti_runtime.o", runtime_tmp_dir);
  src_path_ = fmt::format("{}/_rti_runtime.c", runtime_tmp_dir);

  std::ofstream(src_path_) << header << "\n" << source;
  TI_DEBUG("[cc] compiling runtime -> [{}]:\n{}\n", obj_path_, source);
  execute(cc_program_impl_->config->cc_compile_cmd, obj_path_, src_path_);
}

void CCKernel::launch(RuntimeContext *ctx) {
  if (!kernel_->is_evaluator)
    ActionRecorder::get_instance().record("launch_kernel",
                                          {
                                              ActionArg("kernel_name", name_),
                                          });

  cc_program_impl_->relink();
  TI_TRACE("[cc] entering kernel [{}]", name_);
  auto entry = cc_program_impl_->load_kernel(name_);
  TI_ASSERT(entry);
  auto *context = cc_program_impl_->update_context(ctx);
  (*entry)(context);
  cc_program_impl_->context_to_result_buffer();
  TI_TRACE("[cc] leaving kernel [{}]", name_);
}

size_t CCLayout::compile() {
  ActionRecorder::get_instance().record("compile_layout",
                                        {
                                            ActionArg("layout_source", source),
                                        });

  obj_path_ = fmt::format("{}/_rti_root.o", runtime_tmp_dir);
  src_path_ = fmt::format("{}/_rti_root.c", runtime_tmp_dir);
  auto dll_path = fmt::format("{}/libti_roottest.so", runtime_tmp_dir);

  std::ofstream(src_path_) << cc_program_impl_->get_runtime()->header << "\n"
                           << source << "\n"
                           << "void *Ti_get_root_size(void) { \n"
                           << "  return (void *) sizeof(struct Ti_S0root);\n"
                           << "}\n";

  TI_DEBUG("[cc] compiling root struct -> [{}]:\n{}\n", obj_path_, source);
  execute(cc_program_impl_->config->cc_compile_cmd, obj_path_, src_path_);

  TI_DEBUG("[cc] linking root struct object [{}] -> [{}]", obj_path_, dll_path);
  execute(cc_program_impl_->config->cc_link_cmd, dll_path, obj_path_);

  TI_DEBUG("[cc] loading root struct object: {}", dll_path);
  DynamicLoader dll(dll_path);
  TI_ASSERT_INFO(dll.loaded(), "[cc] could not load shared object: {}",
                 dll_path);

  using FuncGetRootSizeType = size_t();
  auto get_root_size = reinterpret_cast<FuncGetRootSizeType *>(
      dll.load_function("Ti_get_root_size"));
  TI_ASSERT(get_root_size);
  return (*get_root_size)();
}

void CCProgramImpl::relink() {
  if (!need_relink_)
    return;

  dll_path_ = fmt::format("{}/libti_program.so", runtime_tmp_dir);

  std::vector<std::string> objects;
  objects.push_back(runtime_->get_object());
  for (auto const &ker : kernels_) {
    objects.push_back(ker->get_object());
  }

  TI_DEBUG("[cc] linking shared object [{}] with [{}]", dll_path_,
           fmt::join(objects, "] ["));
  execute(this->config->cc_link_cmd, dll_path_, fmt::join(objects, "' '"));

  dll_ = nullptr;
  TI_DEBUG("[cc] loading shared object: {}", dll_path_);
  dll_ = std::make_unique<DynamicLoader>(dll_path_);
  TI_ASSERT_INFO(dll_->loaded(), "[cc] could not load shared object: {}",
                 dll_path_);

  need_relink_ = false;
}

CCFuncEntryType *CCProgramImpl::load_kernel(std::string const &name) {
  return reinterpret_cast<CCFuncEntryType *>(dll_->load_function("Tk_" + name));
}

CCContext *CCProgramImpl::update_context(RuntimeContext *ctx) {
  // TODO(k-ye): Do you have other zero-copy ideas for arg buf?
  std::memcpy(context_->args, ctx->args, taichi_max_num_args * sizeof(uint64));
  context_->earg = (int *)ctx->extra_args;
  return context_.get();
}

void CCProgramImpl::context_to_result_buffer() {
  TI_ASSERT(result_buffer_);
  std::memcpy(result_buffer_, context_->args,
              taichi_max_num_ret_value * sizeof(uint64));
  context_->earg = nullptr;
}

namespace cccp {
bool is_c_backend_available() {
  return true;
}
};  // namespace cccp

TLANG_NAMESPACE_END
