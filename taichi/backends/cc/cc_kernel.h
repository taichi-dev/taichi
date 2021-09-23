#pragma once

#include "taichi/lang_util.h"
#include <set>

TLANG_NAMESPACE_BEGIN

class Kernel;

class CCProgramImpl;

namespace cccp {

class CCKernel {
 public:
  CCKernel(CCProgramImpl *cc_program_impl,
           Kernel *kernel,
           std::string const &source,
           std::string const &name)
      : cc_program_impl(cc_program_impl),
        kernel(kernel),
        name(name),
        source(source) {
  }

  void compile();
  void launch(Context *ctx);
  std::string get_object() {
    return obj_path;
  }

 private:
  CCProgramImpl *cc_program_impl;
  Kernel *kernel;

  std::string name;
  std::string source;

  std::string src_path;
  std::string obj_path;
};

}  // namespace cccp
TLANG_NAMESPACE_END
