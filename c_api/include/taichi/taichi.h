#pragma once

#include "taichi/taichi_platform.h"

#include "taichi/taichi_core.h"

#if TI_WITH_VULKAN
#define VK_NO_PROTOTYPES 1
#include "taichi/taichi_vulkan.h"
#endif  // TI_WITH_VULKAN

#if TI_WITH_OPENGL
#include "taichi/taichi_opengl.h"
#endif
