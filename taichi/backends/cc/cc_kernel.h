#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCKernel {
 public:
  CCKernel(std::string const &source);

  std::string source;
};

}  // namespace cccp
TLANG_NAMESPACE_END
