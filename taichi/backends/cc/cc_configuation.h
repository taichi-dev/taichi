#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

struct CCConfiguation {
  std::string compile_cmd;

  CCConfiguation() : compile_cmd("gcc -shared -fPIC -o '{}' '{}'") {
  }
};

extern CCConfiguation cfg;

bool is_c_backend_available();

}  // namespace cccp
TLANG_NAMESPACE_END
