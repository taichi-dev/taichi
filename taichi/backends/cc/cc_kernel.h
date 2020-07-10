#pragma once

#include "taichi/lang_util.h"
#include <set>

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCProgram;

class CCKernel {
 public:
  CCKernel(CCProgram *program,
           std::string const &source,
           std::string const &name)
      : program(program), name(name), source(source) {
  }

  void compile();
  void launch(Context *ctx);
  std::string get_object() {
    return obj_path;
  }

 private:
  CCProgram *program;

  std::string name;
  std::string source;

  std::string src_path;
  std::string obj_path;
};

}  // namespace cccp
TLANG_NAMESPACE_END
