#pragma once

#include "taichi/taichi_platform.h"

#include "taichi/taichi_core.h"

#ifdef TI_WITH_VULKAN
#include "taichi/taichi_vulkan.h"
#endif  // TI_WITH_VULKAN

#ifdef TI_WITH_OPENGL
#include "taichi/taichi_opengl.h"
#endif  // TI_WITH_OPENGL

#ifdef TI_WITH_CUDA
#include "taichi/taichi_cuda.h"
#endif  // TI_WITH_CUDA

#ifdef TI_WITH_CPU
#include "taichi/taichi_cpu.h"
#endif  // TI_WITH_CPU
