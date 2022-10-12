#pragma once

#include "taichi/taichi_platform.h"

#include "taichi/taichi_core.h"

#ifdef TI_WITH_VULKAN
#ifndef TI_NO_VULKAN_INCLUDES
#include <vulkan/vulkan.h>
#endif  // TI_NO_VULKAN_INCLUDES

#include "taichi/taichi_vulkan.h"
#endif  // TI_WITH_VULKAN

#ifdef TI_WITH_OPENGL
#ifndef TI_NO_OPENGL_INCLUDES
#include <GL/gl.h>
#endif  // TI_NO_OPENGL_INCLUDES

#include "taichi/taichi_opengl.h"
#endif  // TI_WITH_OPENGL
