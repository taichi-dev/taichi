#include "taichi/common/core.h"
#include "cc_launcher.h"
#include "cc_configuation.h"
#include "cc_kernel.h"
#include <fstream>
#include <cstdlib>

TLANG_NAMESPACE_BEGIN
namespace cccp {

CCConfiguation cfg;

template <typename... Args>
int execute(std::string fmt, Args &&... args) {
  auto cmd = fmt::format(fmt, std::forward<Args>(args)...);
  TI_INFO("Executing command: {}", cmd);
  int ret = std::system(cmd.c_str());
  TI_INFO("Command exit status: {}", ret);
  return ret;
}

CCKernel::CCKernel(std::string const &source)
    : source(source) {
  auto dst_path = "/tmp/a.out";
  auto src_path = "/tmp/out.c";
  std::ofstream(src_path) << source;
  execute(cfg.compile_cmd, dst_path, src_path);
  execute(cfg.execute_cmd, dst_path);
}

void CCLauncher::launch(CCKernel *kernel, Context *ctx) {
  TI_INFO("[cc] launching kernel source:\n{}\n", kernel->source);
}

void CCLauncher::keep(std::unique_ptr<CCKernel> kernel) {
  kept_kernels.push_back(std::move(kernel));
}

CCLauncher::CCLauncher() {
}

CCLauncher::~CCLauncher() {
}

bool is_c_backend_available() {
  return true;
}

}  // namespace cccp
TLANG_NAMESPACE_END

