// vim: ft=c
// clang-format off
#include "taichi/util/macros.h"
"#include <stdio.h>\n"
"#include <stdlib.h>\n"
"#include <math.h>\n"
"\n" STR(

typedef char RTi_i8;
typedef short RTi_i16;
typedef int RTi_i32;
typedef long long RTi_i64;
typedef unsigned char RTi_u8;
typedef unsigned short RTi_u16;
typedef unsigned int RTi_u32;
typedef unsigned long long RTi_u64;
typedef float RTi_f32;
typedef double RTi_f64;

) "\n" STR(

/* libc doesn't provide max/min for integers... sadly */
static inline RTi_i32 RTi_max(RTi_i32 x, RTi_i32 y) {
  return x > y ? x : y;
}
static inline RTi_i32 RTi_min(RTi_i32 x, RTi_i32 y) {
  return x < y ? x : y;
}
static inline RTi_i64 RTi_llmax(RTi_i64 x, RTi_i64 y) {
  return x > y ? x : y;
}
static inline RTi_i64 RTi_llmin(RTi_i64 x, RTi_i64 y) {
  return x < y ? x : y;
}

) "\n" STR(

static inline RTi_i32 RTi_rand_i32(void) {
  return mrand48();  // includes negative
}

static inline RTi_i64 RTi_rand_i64(void) {
  return (RTi_i64) (mrand48() << 32) | mrand48();
}

static inline RTi_f64 RTi_rand_f64(void) {
  return drand48();  // [0.0, 1.0)
}

static inline RTi_f32 RTi_rand_f32(void) {
  return (RTi_f32) drand48();  // [0.0, 1.0)
}

)

#define _CC_INSIDE_KERNEL
#include "taichi/backends/cc/context.h"
#undef _CC_INSIDE_KERNEL
