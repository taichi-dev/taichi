#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCProgram;

class CCRuntime {
 public:
  CCRuntime(CCProgram *program,
            std::string const &header)
      : header(header), program(program) {
  }

  std::string header;

 private:
  [[maybe_unused]] CCProgram *program;
};

}  // namespace cccp
TLANG_NAMESPACE_END
