#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCRuntime {
 public:
  CCRuntime(std::string const &source,
            std::string const &header)
    : name(name), source(source), header(header) {
  }

  std::string get_object() {
    return obj_path;
  }

  void compile();

  std::string source;
  std::string header;

 private:
  std::string src_path;
  std::string obj_path;
};

}  // namespace cccp
TLANG_NAMESPACE_END
