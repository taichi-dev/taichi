#include "taichi/common/core.h"
#include "taichi/system/dynamic_loader.h"
#include "cc_program.h"
#include "cc_configuation.h"
#include "cc_runtime.h"
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
  TI_DEBUG("[cc] compiling [{}] -> [{}]:\n{}\n", name, obj_path, source);
  execute(cfg.compile_cmd, obj_path, src_path);
}

void CCKernel::launch(Context *ctx) {
  program->relink();
  auto entry = program->load_kernel(name);
  TI_TRACE("[cc] entering kernel [{}]", name);
  (*entry)();
  TI_TRACE("[cc] leaving kernel [{}]", name);
}

void CCLayout::compile() {
  obj_path = fmt::format("{}/_root.o", runtime_tmp_dir);
  src_path = fmt::format("{}/_root.c", runtime_tmp_dir);

  std::ofstream(src_path) << source
                          << "\n\nstruct S0root *_RTi_get_root() {\n"
                          << "\tstatic struct S0root r;\n\treturn &r;\n}\n";
  TI_DEBUG("[cc] compiling root struct -> [{}]:\n{}\n", obj_path, source);
  execute(cfg.compile_cmd, obj_path, src_path);
}

void CCRuntime::compile() {
  obj_path = fmt::format("{}/_rt_{}.o", runtime_tmp_dir, name);
  src_path = fmt::format("{}/_rt_{}.c", runtime_tmp_dir, name);

  std::ofstream(src_path) << source;
  TI_DEBUG("[cc] compiling runtime [{}] -> [{}]:\n{}\n", name, obj_path, source);
  execute(cfg.compile_cmd, obj_path, src_path);
}

void CCProgram::relink() {
  if (!need_relink)
    return;

  dll_path = fmt::format("{}/libti_program.so", runtime_tmp_dir);

  std::vector<std::string> objects;
  objects.push_back(layout->get_object());
  for (auto const &[name, rt] : runtimes) {
    objects.push_back(rt->get_object());
  }
  for (auto const &[name, ker] : kernels) {
    objects.push_back(ker->get_object());
  }

  TI_DEBUG("[cc] linking shared object [{}] with [{}]", dll_path,
           fmt::join(objects, "] ["));
  execute(cfg.link_cmd, dll_path, fmt::join(objects, "' '"));

  TI_DEBUG("[cc] loading shared object: {}", dll_path);
  dll = std::make_unique<DynamicLoader>(dll_path);
  TI_ASSERT_INFO(dll->loaded(), "[cc] could not load shared object: {}",
                 dll_path);

  need_relink = false;
}

void CCProgram::add_kernel(std::unique_ptr<CCKernel> kernel) {
  TI_ASSERT_INFO(kernels.find(kernel->name) == kernels.end(),
          "Kernel name already exists: {}", kernel->name);
  kernels[kernel->name] = std::move(kernel);
  need_relink = true;
}

void CCProgram::add_runtime(std::unique_ptr<CCRuntime> runtime) {
  TI_ASSERT_INFO(runtimes.find(runtime->name) == runtimes.end(),
          "Runtime module already exists: {}", runtime->name);
  runtimes[runtime->name] = std::move(runtime);
  need_relink = true;
}

void CCProgram::import_runtime(std::string const &name) {
  static std::map<std::string, std::string> src_table;
  if (src_table.empty()) {
    src_table["base"] =
#include "runtime/base.c"
    ;
  }
  TI_DEBUG("[cc] importing runtime module [{}]", name);
  auto it = src_table.find(name);
  TI_ASSERT_INFO(it != src_table.end(), "Runtime module not found: {}", name);
  auto rt = std::make_unique<CCRuntime>(name, it->second);
  add_runtime(std::move(rt));
}

CCFuncEntryType *CCProgram::load_kernel(std::string const &name) {
  return reinterpret_cast<CCFuncEntryType *>(
      dll->load_function("Ti_" + name));
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
