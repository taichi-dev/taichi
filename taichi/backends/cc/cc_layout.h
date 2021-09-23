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
    return obj_path;
  }

  size_t compile();

  std::string source;

 private:
  CCProgramImpl *cc_program_impl_;

  std::string src_path;
  std::string obj_path;
};

}  // namespace cccp
TLANG_NAMESPACE_END
