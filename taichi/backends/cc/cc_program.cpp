#include "taichi/common/core.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/system/dynamic_loader.h"
#include "taichi/util/action_recorder.h"
#include "struct_cc.h"
#include "cc_program.h"
#include "cc_runtime.h"
#include "cc_kernel.h"
#include "cc_layout.h"
#include "cc_utils.h"
#include "context.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

void CCKernel::compile() {
  if (!kernel->is_evaluator)
    ActionRecorder::get_instance().record(
        "compile_kernel", {
                              ActionArg("kernel_name", name),
                              ActionArg("kernel_source", source),
                          });

  obj_path = fmt::format("{}/{}.o", runtime_tmp_dir, name);
  src_path = fmt::format("{}/{}.c", runtime_tmp_dir, name);

  std::ofstream(src_path) << program->get_runtime()->header << "\n"
                          << program->get_layout()->source << "\n"
                          << source;
  TI_DEBUG("[cc] compiling [{}] -> [{}]:\n{}\n", name, obj_path, source);
  execute(program->program->config.cc_compile_cmd, obj_path, src_path);
}

void CCKernel::launch(Context *ctx) {
  if (!kernel->is_evaluator)
    ActionRecorder::get_instance().record("launch_kernel",
                                          {
                                              ActionArg("kernel_name", name),
                                          });

  program->relink();
  TI_TRACE("[cc] entering kernel [{}]", name);
  auto entry = program->load_kernel(name);
  TI_ASSERT(entry);
  auto *context = program->update_context(ctx);
  (*entry)(context);
  program->context_to_result_buffer();
  TI_TRACE("[cc] leaving kernel [{}]", name);
}

size_t CCLayout::compile() {
  ActionRecorder::get_instance().record("compile_layout",
                                        {
                                            ActionArg("layout_source", source),
                                        });

  obj_path = fmt::format("{}/_rti_root.o", runtime_tmp_dir);
  src_path = fmt::format("{}/_rti_root.c", runtime_tmp_dir);
  auto dll_path = fmt::format("{}/libti_roottest.so", runtime_tmp_dir);

  std::ofstream(src_path) << program->get_runtime()->header << "\n"
                          << source << "\n"
                          << "void *Ti_get_root_size(void) { \n"
                          << "  return (void *) sizeof(struct Ti_S0root);\n"
                          << "}\n";

  TI_DEBUG("[cc] compiling root struct -> [{}]:\n{}\n", obj_path, source);
  execute(program->program->config.cc_compile_cmd, obj_path, src_path);

  TI_DEBUG("[cc] linking root struct object [{}] -> [{}]", obj_path, dll_path);
  execute(program->program->config.cc_link_cmd, dll_path, obj_path);

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

void CCRuntime::compile() {
  ActionRecorder::get_instance().record("compile_runtime",
                                        {
                                            ActionArg("runtime_header", header),
                                            ActionArg("runtime_source", source),
                                        });

  obj_path = fmt::format("{}/_rti_runtime.o", runtime_tmp_dir);
  src_path = fmt::format("{}/_rti_runtime.c", runtime_tmp_dir);

  std::ofstream(src_path) << header << "\n" << source;
  TI_DEBUG("[cc] compiling runtime -> [{}]:\n{}\n", obj_path, source);
  execute(program->program->config.cc_compile_cmd, obj_path, src_path);
}

void CCProgram::relink() {
  if (!need_relink)
    return;

  dll_path = fmt::format("{}/libti_program.so", runtime_tmp_dir);

  std::vector<std::string> objects;
  objects.push_back(runtime->get_object());
  for (auto const &ker : kernels) {
    objects.push_back(ker->get_object());
  }

  TI_DEBUG("[cc] linking shared object [{}] with [{}]", dll_path,
           fmt::join(objects, "] ["));
  execute(program->config.cc_link_cmd, dll_path, fmt::join(objects, "' '"));

  dll = nullptr;
  TI_DEBUG("[cc] loading shared object: {}", dll_path);
  dll = std::make_unique<DynamicLoader>(dll_path);
  TI_ASSERT_INFO(dll->loaded(), "[cc] could not load shared object: {}",
                 dll_path);

  need_relink = false;
}

void CCProgram::compile_layout(SNode *root) {
  CCLayoutGen gen(this, root);
  layout = gen.compile();
  size_t root_size = layout->compile();
  size_t gtmp_size = taichi_global_tmp_buffer_size;
  size_t args_size = taichi_max_num_args * sizeof(uint64);

  TI_INFO("[cc] C backend root buffer size: {} B", root_size);

  ActionRecorder::get_instance().record(
      "allocate_buffer", {
                             ActionArg("root_size", (int32)root_size),
                             ActionArg("gtmp_size", (int32)gtmp_size),
                         });

  root_buf.resize(root_size, 0);
  gtmp_buf.resize(gtmp_size, 0);
  args_buf.resize(args_size, 0);

  context->root = root_buf.data();
  context->gtmp = gtmp_buf.data();
  context->args = (uint64 *)args_buf.data();
  context->earg = nullptr;
}

void CCProgram::add_kernel(std::unique_ptr<CCKernel> kernel) {
  kernels.push_back(std::move(kernel));
  need_relink = true;
}

void CCProgram::init_runtime() {
  runtime = std::make_unique<CCRuntime>(this,
#include "runtime/base.h"
                                        "\n",
#include "runtime/base.c"
                                        "\n");
  runtime->compile();
}

CCFuncEntryType *CCProgram::load_kernel(std::string const &name) {
  return reinterpret_cast<CCFuncEntryType *>(dll->load_function("Tk_" + name));
}

CCProgram::CCProgram(Program *program) : program(program) {
  init_runtime();

  context = std::make_unique<CCContext>();
}

CCContext *CCProgram::update_context(Context *ctx) {
  // TODO(k-ye): Do you have other zero-copy ideas for arg buf?
  std::memcpy(context->args, ctx->args, taichi_max_num_args * sizeof(uint64));
  context->earg = (int *)ctx->extra_args;
  return context.get();
}

void CCProgram::context_to_result_buffer() {
  TI_ASSERT(program->result_buffer);
  std::memcpy(program->result_buffer, context->args,
              sizeof(uint64));  // XXX: assumed 1 return
  context->earg = nullptr;
}

CCProgram::~CCProgram() {
}

bool is_c_backend_available() {
  return true;
}

}  // namespace cccp
TLANG_NAMESPACE_END
