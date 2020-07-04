#pragma once

#include "taichi/lang_util.h"
#include <map>

TI_NAMESPACE_BEGIN
class DynamicLoader;
TI_NAMESPACE_END

TLANG_NAMESPACE_BEGIN

namespace cccp {

class CCKernel;
class CCLayout;
class CCRuntime;

using CCFuncEntryType = void();

class CCProgram {
  // Launch C compiler to compile generated source code, and run them
 public:
  CCProgram();
  ~CCProgram();

  void add_kernel(std::unique_ptr<CCKernel> kernel);
  void add_runtime(std::unique_ptr<CCRuntime> runtime);
  CCFuncEntryType *load_kernel(std::string const &name);
  void import_runtime(std::string const &name);
  void relink();

  std::map<std::string, std::unique_ptr<CCKernel>> kernels;
  std::map<std::string, std::unique_ptr<CCRuntime>> runtimes;
  std::unique_ptr<CCLayout> layout;
  std::unique_ptr<DynamicLoader> dll;
  std::string dll_path;
  bool need_relink{true};
};

}  // namespace cccp
TLANG_NAMESPACE_END
