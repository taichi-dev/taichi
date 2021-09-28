#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
class CCProgramImpl;

namespace cccp {

class CCRuntime {
 public:
  CCRuntime(CCProgramImpl *cc_program_impl,
            std::string const &header,
            std::string const &source)
      : header(header), source(source), cc_program_impl_(cc_program_impl) {
  }

  std::string get_object() {
    return obj_path_;
  }

  void compile();

  std::string header;
  std::string source;

 private:
  CCProgramImpl *cc_program_impl_{nullptr};
  std::string src_path_;
  std::string obj_path_;
};

}  // namespace cccp
TLANG_NAMESPACE_END
