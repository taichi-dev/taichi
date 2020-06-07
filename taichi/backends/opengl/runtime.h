#ifdef __GLSL__
// clang-format off

#include "taichi/util/macros.h"
STR(
layout(shared, binding = 6) buffer runtime {
  int _rand_state_;
  int _msg_count_;
  int _mesg_i32_[];
};
)

// clang-format on
#else

#define MAX_PRINT_ENTRIES (1024 * 8) // * 4 = 32 KB

struct GLSLRuntime {
  int rand_state;
  int msg_count;
  int msg_buf[MAX_PRINT_ENTRIES];
} __attribute__((packed));

#endif
