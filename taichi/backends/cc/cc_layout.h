#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCProgram;

class CCLayout {
 public:
  CCLayout(CCProgram *program) : program(program) {
  }

  std::string get_object() {
    return obj_path;
  }

  size_t compile();

  std::string source;

 private:
  CCProgram *program;

  std::string src_path;
  std::string obj_path;
};

}  // namespace cccp
TLANG_NAMESPACE_END
