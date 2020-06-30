#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCProgram;

class CCKernel {
 public:
  CCKernel(std::string const &source, std::string const &name)
      : source(source), name(name) {
  }

  void compile();
  void launch(CCProgram *program, Context *ctx);

  std::string source;
  std::string name;

  std::string src_path;
  std::string bin_path;
};

}  // namespace cccp
TLANG_NAMESPACE_END
