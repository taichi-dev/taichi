#pragma once

#ifndef ANDROID
#include "GLFW/glfw3.h"
#endif  // ANDROID

namespace taichi::lang::window_system {

bool glfw_context_acquire();
void glfw_context_release();

}
