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
  bin_path = fmt::format("{}/{}.so", runtime_tmp_dir, name);
  src_path = fmt::format("{}/{}.c", runtime_tmp_dir, name);

  std::ofstream(src_path) << source;
  TI_INFO("[cc] compiling kernel [{}]:\n{}\n", name, source);
  execute(cfg.compile_cmd, bin_path, src_path);
}


void CCKernel::launch(CCProgram *launcher, Context *ctx) {
  using FuncEntryType = void();
  DynamicLoader dll(bin_path);
  TI_ASSERT_INFO(dll.loaded(), "[cc] could not load shared object: {}", bin_path);
  auto main = reinterpret_cast<FuncEntryType *>(dll.load_function(get_sym_name(name)));
  TI_INFO("[cc] entering kernel [{}]", name);
  (*main)();
  TI_INFO("[cc] leaving kernel [{}]", name);
  //execute(cfg.execute_cmd, bin_path);
}

void CCProgram::launch(CCKernel *kernel, Context *ctx) {
  kernel->launch(this, ctx);
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

