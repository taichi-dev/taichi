#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCProgram;

class CCKernel {
 public:
  CCKernel(CCProgram *program,
           std::string const &source,
           std::string const &name)
      : name(name), source(source), program(program) {
  }

  void compile();
  void launch(Context *ctx);
  std::string get_object() {
    return obj_path;
  }

  std::string name;
  std::string source;

 private:
  CCProgram *program;

  std::string src_path;
  std::string obj_path;
};

}  // namespace cccp
TLANG_NAMESPACE_END
