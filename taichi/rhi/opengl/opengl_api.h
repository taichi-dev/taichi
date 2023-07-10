#pragma once

#include "taichi/common/core.h"
#include "taichi/rhi/impl_support.h"
#include "taichi/rhi/device.h"

namespace taichi::lang {
namespace opengl {

void set_gles_override(bool value);
void unset_gles_override();
bool initialize_opengl(bool use_gles = false, bool error_tolerance = false);
bool is_opengl_api_available(bool use_gles = false);
bool is_gles();
void reset_opengl();

std::shared_ptr<Device> make_opengl_device();

#define PER_OPENGL_EXTENSION(x) extern bool opengl_extension_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

}  // namespace opengl
}  // namespace taichi::lang
