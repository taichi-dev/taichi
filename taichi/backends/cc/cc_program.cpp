#include "taichi/common/core.h"
#include "taichi/system/dynamic_loader.h"
#include "cc_program.h"
#include "cc_configuation.h"
#include "cc_kernel.h"
#include "cc_layout.h"
#include "cc_utils.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

CCConfiguation cfg;

void CCKernel::compile() {
  obj_path = fmt::format("{}/{}.o", runtime_tmp_dir, name);
  src_path = fmt::format("{}/{}.c", runtime_tmp_dir, name);

  std::ofstream(src_path) << source;
  TI_INFO("[cc] compiling kernel [{}]:\n{}\n", name, source);
  execute(cfg.compile_cmd, obj_path, src_path);
}

void CCKernel::launch(Context *ctx) {
  program->relink();
  auto entry = program->load_kernel(name);
  TI_INFO("[cc] entering kernel [{}]", name);
  (*entry)();
  TI_INFO("[cc] leaving kernel [{}]", name);
}

void CCLayout::compile() {
  obj_path = fmt::format("{}/_root.o", runtime_tmp_dir);
  src_path = fmt::format("{}/_root.c", runtime_tmp_dir);

  std::ofstream(src_path) << source;
  TI_INFO("[cc] compiling root struct [{}]:\n{}\n", name, source);
  execute(cfg.compile_cmd, obj_path, src_path);
}

void CCProgram::relink(Context *ctx) {
  if (!need_relink)
    return;

  dll_path = fmt::format("{}/libthis_program.so", runtime_tmp_dir);

  std::vector<std::string> objects;
  objects.push_back(layout->get_object());
  for (auto const &ker: kernels) {
    objects.push_back(ker->get_object());
  }

  TI_INFO("[cc] linking program [{}] with [{}]", dll_path, fmt::join("] [", objects));
  execute(cfg.link_cmd, dll_path, fmt::join(" ", objects));

  TI_INFO("[cc] loading program: {}", dll_path);
  dll = std::make_unique<DynamicLoader>(dll_path);
  TI_ASSERT_INFO(dll->loaded(), "[cc] could not load shared object: {}", dll_path);

  need_relink = false;
}

void CCProgram::add_kernel(std::unique_ptr<CCKernel> kernel) {
  kernels.push_back(std::move(kernel));
  need_relink = true;
}

CCFuncEntryType *CCProgram::load_kernel(std::string const &name) {
  return reinterpret_cast<CCFuncEntryType *>(program->dll->load_function(get_func_sym(name)));
}

CCProgram::CCProgram() {
}

CCProgram::~CCProgram() {
}

bool is_c_backend_available() {
  return true;
}

}  // namespace cccp
TLANG_NAMESPACE_END

