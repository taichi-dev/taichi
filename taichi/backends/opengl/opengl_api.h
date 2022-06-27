#pragma once

#include "taichi/common/core.h"
#include "taichi/backends/device.h"

namespace taichi {
namespace lang {
namespace opengl {

bool initialize_opengl(bool use_gles = false, bool error_tolerance = false);
bool is_opengl_api_available(bool use_gles = false);
bool is_gles();

std::shared_ptr<Device> make_opengl_device();

#define PER_OPENGL_EXTENSION(x) extern bool opengl_extension_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
