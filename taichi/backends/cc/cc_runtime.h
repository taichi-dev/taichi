#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCRuntime {
 public:
  CCRuntime(std::string const &name,
            std::string const &source = "")
    : name(name), source(source) {
  }

  std::string get_object() {
    return obj_path;
  }

  void compile();

  std::string name;
  std::string source;

 private:
  std::string src_path;
  std::string obj_path;
};

}  // namespace cccp
TLANG_NAMESPACE_END
