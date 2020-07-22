#include "taichi/common/core.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/system/dynamic_loader.h"
#include "taichi/util/action_recorder.h"
#include "struct_cc.h"
#include "cc_program.h"
#include "cc_configuation.h"
#include "cc_runtime.h"
#include "cc_kernel.h"
#include "cc_layout.h"
#include "cc_utils.h"
#include "context.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

CCConfiguation cfg;

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
  execute(cfg.compile_cmd, obj_path, src_path);
}

CCContext::CCContext(CCProgram *program, Context *ctx)
    : args(ctx->args), earg((int *)ctx->extra_args) {
  root = program->get_root_buffer();
  gtmp = program->get_gtmp_buffer();
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
  CCContext cc_ctx(program, ctx);
  (*entry)(&cc_ctx);
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
  execute(cfg.compile_cmd, obj_path, src_path);

  TI_DEBUG("[cc] linking root struct object [{}] -> [{}]", obj_path, dll_path);
  execute(cfg.link_cmd, dll_path, obj_path);

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
  execute(cfg.compile_cmd, obj_path, src_path);
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
  execute(cfg.link_cmd, dll_path, fmt::join(objects, "' '"));

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

  TI_INFO("[cc] C backend root buffer size: {} B", root_size);

  ActionRecorder::get_instance().record(
      "allocate_buffer", {
                             ActionArg("root_size", (int32)root_size),
                             ActionArg("gtmp_size", (int32)gtmp_size),
                         });

  root_buf.resize(root_size, 0);
  gtmp_buf.resize(gtmp_size, 0);
}

void CCProgram::add_kernel(std::unique_ptr<CCKernel> kernel) {
  kernels.push_back(std::move(kernel));
  need_relink = true;
}

// TODO: move this to cc_runtime.cpp:
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

CCProgram::CCProgram() {
  init_runtime();
}

CCProgram::~CCProgram() {
}

bool is_c_backend_available() {
  return true;
}

}  // namespace cccp
TLANG_NAMESPACE_END
