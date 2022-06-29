#ifdef _CC_INSIDE_KERNEL
#include "taichi/util/macros.h"
// clang-format off
STR(
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
  struct Ti_S0root *root;
  // In some C compilers `void *p; p + 1 == p;`, so let's use `char *p`:
  Ti_i8 *gtmp;

  union Ti_BitCast *args;
  int *earg;
};
)

// clang-format on
#else  // _CC_INSIDE_KERNEL

#include "cc_program.h"
#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN

namespace cccp {

struct CCContext {
  void *root;
  void *gtmp;

  uint64_t *args;
  int *earg;
};

};  // namespace cccp

TLANG_NAMESPACE_END

#endif  // _CC_INSIDE_KERNEL
