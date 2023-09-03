#pragma once

#ifdef TI_WITH_GLFW
#include "GLFW/glfw3.h"
#ifdef TI_WITH_METAL
#define GLFW_EXPOSE_NATIVE_COCOA
#import <GLFW/glfw3native.h>
#endif
#endif  // TI_WITH_GLFW

namespace taichi::lang::window_system {

bool glfw_context_acquire();
void glfw_context_release();

}  // namespace taichi::lang::window_system
