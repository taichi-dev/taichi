#ifdef _CC_INSIDE_KERNEL
#include "taichi/util/macros.h"
STR(
union RTi_BitCast {
  RTi_i64 val_i64;
  RTi_i32 val_i32;
  RTi_i16 val_i16;
  RTi_i8 val_i8;
  RTi_u64 val_u64;
  RTi_u32 val_u32;
  RTi_u16 val_u16;
  RTi_u8 val_u8;
  RTi_f32 val_f32;
  RTi_f64 val_f64;
  RTi_i64 *ptr_i64;
  RTi_i32 *ptr_i32;
  RTi_i16 *ptr_i16;
  RTi_i8 *ptr_i8;
  RTi_u64 *ptr_u64;
  RTi_u32 *ptr_u32;
  RTi_u16 *ptr_u16;
  RTi_u8 *ptr_u8;
  RTi_f32 *ptr_f32;
  RTi_f64 *ptr_f64;
};

struct RTi_Context {
  struct S0root *root;
  // In some C compilers `void *p; p + 1 == p;`, so let's use `char *p`:
  RTi_i8 *gtmp;

  union RTi_BitCast *args;
  int *earg;
};
)

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

  CCContext(CCProgram *program, Context *ctx);
};

};

TLANG_NAMESPACE_END

#endif // _CC_INSIDE_KERNEL
