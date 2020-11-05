// vim: ft=opencl
// clang-format off
#include "taichi/util/macros.h"
STR(

typedef char Ti_i8;
typedef short Ti_i16;
typedef int Ti_i32;
typedef long long Ti_i64;
typedef unsigned char Ti_u8;
typedef unsigned short Ti_u16;
typedef unsigned int Ti_u32;
typedef unsigned long long Ti_u64;
typedef float Ti_f32;
typedef double Ti_f64;

{
}

) "\n" STR(
)

#define _TI_OPENCL_INSIDE_KERNEL
#include "taichi/backends/opencl/context.h"
#undef _TI_OPENCL_INSIDE_KERNEL

"\n" STR(

__global struct Ti_Context ti_ctx;
__global struct Ti_S0root ti_root;

)
