#ifdef TI_INSIDE_METAL_CODEGEN

#ifndef TI_METAL_NESTED_INCLUDE
#define STR2(...) #__VA_ARGS__
#define STR(...) STR2(__VA_ARGS__)
#else
#define STR(...)
#endif  // TI_METAL_NESTED_INCLUDE

#else

#include <cstdint>

#define STR(...) __VA_ARGS__

#define device
#define constant
#define thread
#define kernel

#define byte char

#include "taichi/platform/metal/shaders/atomic_stubs.h"

#endif  // TI_INSIDE_METAL_CODEGEN
