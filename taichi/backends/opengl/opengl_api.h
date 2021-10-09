#pragma once

#include "taichi/common/core.h"
#include "taichi/ir/transforms.h"
#include "taichi/backends/device.h"

#include <string>
#include <vector>
#include <optional>

TLANG_NAMESPACE_BEGIN

class Kernel;
class OffloadedStmt;

namespace opengl {

bool initialize_opengl(bool error_tolerance = false);
bool is_opengl_api_available();

std::unique_ptr<Device> get_opengl_device();

#define PER_OPENGL_EXTENSION(x) extern bool opengl_extension_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

extern int opengl_threads_per_block;

#define TI_OPENGL_REQUIRE(used, x) \
  ([&]() {                         \
    if (opengl_extension_##x) {    \
      used.extension_##x = true;   \
      return true;                 \
    }                              \
    return false;                  \
  })()

}  // namespace opengl

TLANG_NAMESPACE_END
