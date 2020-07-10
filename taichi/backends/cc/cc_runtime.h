#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCRuntime {
 public:
  CCRuntime(std::string const &header, std::string const &source)
      : header(header), source(source) {
  }

  std::string get_object() {
    return obj_path;
  }

  void compile();

  std::string header;
  std::string source;

 private:
  std::string src_path;
  std::string obj_path;
};

}  // namespace cccp
TLANG_NAMESPACE_END
