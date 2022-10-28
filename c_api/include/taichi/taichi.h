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

#ifdef TI_WITH_CUDA
#ifndef TI_NO_CUDA_INCLUDES
// Only a few CUDA types is needed, including the entire <cuda.h> is overkill
// for this
#include <cuda.h>
#endif  // TI_NO_CUDA_INCLUDES

#include "taichi/taichi_cuda.h"
#endif  // TI_WITH_CUDA

#ifdef TI_WITH_CPU
#include "taichi/taichi_cpu.h"
#endif  // TI_WITH_CPU
