#pragma once

#include "taichi/lang_util.h"

class CCProgramImpl;

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCLayout {
 public:
  CCLayout(CCProgramImpl *cc_program_impl) : cc_program_impl_(cc_program_impl) {
  }

  std::string get_object() {
    return obj_path_;
  }

  size_t compile();

  std::string source;

 private:
  CCProgramImpl *cc_program_impl_{nullptr};

  std::string src_path_;
  std::string obj_path_;
};

}  // namespace cccp
TLANG_NAMESPACE_END
