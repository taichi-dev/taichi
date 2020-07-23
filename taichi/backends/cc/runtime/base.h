// vim: ft=c
// clang-format off
#include "taichi/util/macros.h"
"#include <stdio.h>\n"
"#include <stdlib.h>\n"
"#include <math.h>\n"
"\n" STR(

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

) "\n" STR(

/* libc doesn't provide max/min for integers... sadly */
static inline Ti_i32 Ti_max(Ti_i32 x, Ti_i32 y) {
  return x > y ? x : y;
}
static inline Ti_i32 Ti_min(Ti_i32 x, Ti_i32 y) {
  return x < y ? x : y;
}
static inline Ti_i64 Ti_llmax(Ti_i64 x, Ti_i64 y) {
  return x > y ? x : y;
}
static inline Ti_i64 Ti_llmin(Ti_i64 x, Ti_i64 y) {
  return x < y ? x : y;
}

) "\n" STR(

static inline Ti_i32 Ti_rand_i32(void) {
  return mrand48();  // includes negative
}

static inline Ti_i64 Ti_rand_i64(void) {
  return ((Ti_i64) mrand48() << 32) | mrand48();
}

static inline Ti_f64 Ti_rand_f64(void) {
  return drand48();  // [0.0, 1.0)
}

static inline Ti_f32 Ti_rand_f32(void) {
  return (Ti_f32) drand48();  // [0.0, 1.0)
}

)

#define _CC_INSIDE_KERNEL
#include "taichi/backends/cc/context.h"
#undef _CC_INSIDE_KERNEL
