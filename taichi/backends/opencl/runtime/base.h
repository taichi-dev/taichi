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

union Ti_BitCast {
  Ti_i64 val_i64;
  Ti_i32 val_i32;
  Ti_i16 val_i16;
  Ti_i8 val_i8;
  Ti_u64 val_u64;
  Ti_u32 val_u32;
  Ti_u16 val_u16;
  Ti_u8 val_u8;
  Ti_f32 val_f32;
  Ti_f64 val_f64;
  Ti_i64 *ptr_i64;
  Ti_i32 *ptr_i32;
  Ti_i16 *ptr_i16;
  Ti_i8 *ptr_i8;
  Ti_u64 *ptr_u64;
  Ti_u32 *ptr_u32;
  Ti_u16 *ptr_u16;
  Ti_u8 *ptr_u8;
  Ti_f32 *ptr_f32;
  Ti_f64 *ptr_f64;
  void *ptr_void;
};

struct Ti_Context {
  union Ti_BitCast args[8];
  Ti_i32 extra_args[8 * 8];
};

)
