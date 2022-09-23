#pragma once

#include "taichi/util/lang_util.h"
#include <set>

namespace taichi::lang {

class Kernel;

class CCProgramImpl;

namespace cccp {

class CCKernel {
 public:
  CCKernel(CCProgramImpl *cc_program_impl,
           Kernel *kernel,
           std::string const &source,
           std::string const &name)
      : cc_program_impl_(cc_program_impl),
        kernel_(kernel),
        name_(name),
        source_(source) {
  }

  void compile();
  void launch(RuntimeContext *ctx);
  std::string get_object() {
    return obj_path_;
  }

 private:
  CCProgramImpl *cc_program_impl_{nullptr};
  Kernel *kernel_;

  std::string name_;
  std::string source_;

  std::string src_path_;
  std::string obj_path_;
};

}  // namespace cccp
}  // namespace taichi::lang
