#pragma once
#ifndef TAICHI_H
#define TAICHI_H

#include "taichi_platform.h"

#include "taichi_core.h"

#ifdef TI_WITH_VULKAN
#include "taichi_vulkan.h"
#endif  // TI_WITH_VULKAN

#ifdef TI_WITH_OPENGL
#include "taichi_opengl.h"
#endif  // TI_WITH_OPENGL

#ifdef TI_WITH_CUDA
#include "taichi_cuda.h"
#endif  // TI_WITH_CUDA

#ifdef TI_WITH_CPU
#include "taichi_cpu.h"
#endif  // TI_WITH_CPU

#ifdef TI_WITH_METAL
#include "taichi_metal.h"
#endif  // TI_WITH_METAL

#endif  // TAICHI_H
