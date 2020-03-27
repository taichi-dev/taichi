#ifdef TI_INSIDE_METAL_CODEGEN

#ifndef TI_METAL_NESTED_INCLUDE
#define STR2(...) #__VA_ARGS__
#define STR(...) STR2(__VA_ARGS__)
#else
// If we are emitting to Metal source code, and the shader file is included by
// some other shader file, then skip emitting the code for the nested shader,
// otherwise there could be a symbol redefinition error. That is, we only emit
// the source code for the shader being directly included by the host side.
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

#include "taichi/backends/metal/shaders/atomic_stubs.h"

#endif  // TI_INSIDE_METAL_CODEGEN
